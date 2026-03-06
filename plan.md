# Full ERISC0 + ERISC1 Coexistence on All Connected ETH Tiles

## Goal

Every connected ETH core on every chip (MMIO and remote) simultaneously runs:
- **ERISC0**: Metal's fabric router (for op-level CCL communication)
- **ERISC1**: Lite fabric relay (for UMD-level L1 reads/writes to remote devices)

They coexist on the same ETH tile because they use separate L1 regions, separate NOC
TRIDs, and separate TXQs (fabric router: TXQ0/TXQ1, lite fabric: TXQ2).

## Current State (what's already done)

### Dual-channel architecture (Groups 1-5 complete)
- **Ch0**: outbound commands (host→remote writes + read commands)
- **Ch1**: inbound read responses (remote→host)
- 4 buffer slots per channel for pipelining
- Per-TRID NOC barriers: ch0 uses TRIDs 8-11, ch1 uses TRIDs 12-15
- Stream register assignments: ch0 IDs 23-25, ch1 IDs 26-28
- UMD: `HostToLiteFabricInterface` has `recv_ch1` state, `flush_recv_ch1_h2d()`,
  `read_one_page()` uses ch1 receiver buffers
- Memory layout: 56KB (base 0x62000), FW and UMD memory maps in sync

### Lite fabric / fabric router separation
- Lite fabric: TXQ2, TRIDs 8-15, stream regs 23-28, L1 region 0x62000-0x70000
- Fabric router: TXQ0/TXQ1, TRIDs 0-7, stream regs 0-22+29-31, L1 below 0x62000

### N-hop BFS discovery (Phase 2b)
- BFS discovers multi-hop chips, launches downstream tunnels, configures forwarding
- Forwarding stays on ch0 only; direct 1-hop reads use ch1
- `downstream_sender_cores_` tracks which cores have ERISC1 running as downstream senders

---

## Remaining Work

### Phase A: Remove ERISC0 Kill from Lite Fabric FW

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

Currently `main()` (lines 362-382) kills ERISC0 immediately on boot:
```cpp
*reinterpret_cast<volatile uint32_t*>(kSoftResetAddr) = 0x46800;  // ERISC0 in reset
*reinterpret_cast<volatile uint32_t*>(0xFFB90000) = 0x1;          // TXQ0 KEEPALIVE
```

This was needed because:
1. ERISC0 syseng FW from POR uses TXQ0 — conflicts with lite fabric
2. ETH link needs keepalive frames; killing ERISC0 removes its natural keepalive

With fabric router running on ERISC0:
1. Fabric router uses TXQ0/TXQ1 — no conflict with lite fabric's TXQ2
2. Fabric router generates regular ETH traffic — natural keepalive

**Changes:**
- Remove the ERISC0 kill block (lines 362-382)
- Remove TXQ0 KEEPALIVE enable (line 467): `*reinterpret_cast<volatile uint32_t*>(ETH_TXQ0_REGS_START + ETH_TXQ_CTRL) = ETH_TXQ_CTRL_KEEPALIVE;`
  - TXQ0 is now managed by ERISC0's fabric router, not ERISC1
  - Keep TXQ2 enable (line 468) — that's lite fabric's own TXQ
- Remove periodic keepalive in `service_lite_fabric()` (lines 228-234):
  ```cpp
  if ((diag_loop_counter & 0xFFFF) == 0 && diag_loop_counter > 0) { ... }
  ```

**Caveat — boot ordering:** Lite fabric (ERISC1) boots in Phase 2, fabric router
(ERISC0) boots in Phase 4 (init_fabric). There's a window where ERISC1 is running
but ERISC0 hasn't started yet. During this window, there's no ETH keepalive from
ERISC0. Options:
1. **Keep software keepalive until ERISC0 is confirmed running** — check a flag/register
2. **Accept the gap** — the window is short (seconds), BH MAC timeout is ~10s
3. **Start ERISC0 earlier** — move fabric router init before Phase 2b BFS

Recommendation: option 2. The boot window is short and BFS operations provide
ETH traffic anyway. If we see link timeouts during boot, add a conditional keepalive
that checks whether ERISC0 is running (read TXQ0 CTRL register or a shared flag).

### Phase B: Remove Defensive ERISC0 Re-kill

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

Currently (not visible in recent reads but was added previously) there may be a
defensive ERISC0 re-kill every ~128K iterations in `service_lite_fabric()`. With
fabric router running, this would kill it. Remove any such logic.

Also verify no other code path in `lite_fabric.cpp` or `channels.hpp` touches
the soft reset register (0xFFB121B0) or puts ERISC0 in reset.

Search patterns:
```
grep -n "0xFFB121B0\|0x46800\|kSoftResetAddr\|soft_reset\|assert.*risc.*reset" \
  tt_metal/lite_fabric/hw/src/lite_fabric.cpp \
  tt_metal/lite_fabric/hw/inc/channels.hpp
```

### Phase C: Keepalive Simplification

**File: `tt_metal/lite_fabric/hw/inc/channels.hpp`**

The sentinel spin loop in `service_fabric_request` (around line 460) may have
keepalive logic inside. Remove it — fabric router traffic keeps the link alive.

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

Remove the periodic keepalive block in `service_lite_fabric()` (lines 228-234).

### Phase D: Configure Fabric — Launch ERISC0 on ALL Cores

**File: `tt_metal/impl/device/device.cpp` — `Device::configure_fabric()`**

The MMIO section (lines 461-536) already deasserts ERISC0 on all fabric program
ETH cores. This is correct — it handles ERISC0 deassert with ERISC1 kept alive
(soft reset 0x46000). No changes needed here if it already covers all connected cores.

Verify: the fabric program's `logical_cores()` includes ALL connected ETH cores,
not just a subset. If the fabric program only uses some cores, we need to ensure
that those cores include all cores where lite fabric (ERISC1) is running.

**File: `tt_metal/impl/context/metal_context.cpp` — `initialize_remote_eth_cores_for_fabric()`**

Lines 3449-3463: the `erisc1_running` detection currently checks:
1. `peer_is_mmio` — 1-hop cores peering with MMIO
2. Tunnel endpoints — cores in `tunnels_from_mmio`
3. Downstream senders — cores in `downstream_sender_cores_`

With full coexistence, ALL connected ETH cores on ALL remote chips have ERISC1
running (launched in Phase 2 for MMIO-peering cores, Phase 2b BFS for everything
else). The current detection logic may miss some cores.

**Change:** Simplify to `erisc1_running = true` for all cores where lite fabric is
active. Since we launch lite fabric on all connected MMIO ETH cores (Phase 2) and
their neighbors propagate it via init-fsm (Phase 2/2b), all connected remote ETH
cores have ERISC1. Set `erisc1_running = true` unconditionally when `lite_fabric_hal_`
is present.

The soft reset values:
- `0x46000`: ERISC0 out of reset + ERISC1 out of reset + bits 13/14/18
- `0x47000`: ERISC0 out of reset + ERISC1 in reset + bits 13/14/18

With full coexistence, always use `0x46000` (both running).

### Phase E: Update Lite Fabric Bindings

**File: `tt_metal/impl/context/metal_context.cpp` — `update_lite_fabric_bindings_for_fabric_routers()`**

Currently (lines 2459-2538) this rebinds 1-hop chips to channels where the fabric
router's remote peer has been launched. It skips chips with forwarding chains.

With full coexistence, ALL ETH cores have both ERISC0 and ERISC1 running. The
binding logic should still work because:
- Forwarding channels are still excluded (multi-hop chains must not be disturbed)
- `remote_fabric_eth_channels_` tracks cores where fabric router was launched

Verify this function doesn't break when ALL cores have fabric routers. The current
filtering (skip forwarding channels, only include cores in `remote_fabric_eth_channels_`)
should be sufficient.

### Phase F: TXQ0 Boot Ordering

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

After removing the ERISC0 kill, the TXQ0 self-healing check (lines 425-463) needs
adjustment. Currently it:
1. Checks TXQ2 status and recovers if stuck
2. Enables TXQ0 KEEPALIVE for init handshake

With ERISC0 running fabric router:
- TXQ0 is managed by ERISC0 — ERISC1 must NOT touch it
- Remove TXQ0 CTRL write (line 467)
- Keep TXQ2 self-healing (lines 425-451) — that's lite fabric's own TXQ
- The init handshake (`ConnectedRiscInterface` in `risc_interface.hpp`) currently
  uses TXQ0 for `eth_write_remote_reg`. This conflicts with fabric router's TXQ0.

**Resolution:** The init handshake only runs during Phase 2 boot, before fabric
router starts. Once routing_init completes, TXQ0 is not used by lite fabric again.
The fabric router starts later (Phase 4). So there's no actual TXQ0 contention
during normal operation — the concern is only if lite fabric restarts while
fabric router is running (which doesn't happen in normal flow).

If we want to be safe: change ConnectedRiscInterface to use TXQ2 for the init
handshake as well. But this is low priority since the timing doesn't overlap.

### Phase G: TXQ Diagnostic on Non-MMIO Cores

Lines 454-462 in `main()` send a diagnostic breadcrumb to the MMIO side via TXQ0:
```cpp
if (!structs->config.is_mmio && !cmd_ongoing) {
    internal_::eth_send_packet<false>(0, src_addr >> 4, dst_addr >> 4, 1);
}
```

This uses TXQ0 which will be used by fabric router. Remove or change to TXQ2.

### Phase H: D-Cache Optimization (Nice-to-Have)

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

The `noc_self_read_word()` calls in `service_lite_fabric()` (mailbox polling, lines
178-215) and `object_init()` (is_mmio check, lines 322-328) are slow because they
do a full NOC DMA read to bypass the D-cache.

Optimization ideas:
1. **Stream register notification**: Host writes to a stream register instead of L1.
   Stream registers bypass D-cache. Lite fabric reads the stream register directly.
2. **RISC-V CSR uncacheable region**: Mark the forwarding config region as uncacheable
   via CSR 0x7c0 settings.
3. **RISC-V volatile + invalidate_l1_cache()**: The `invalidate_l1_cache()` at the
   top of `service_lite_fabric()` should flush stale D-cache lines. Verify this
   works for the specific L1 addresses being polled.

Deferred until coexistence is working.

### Phase I: Multi-Path Tunnels (Nice-to-Have)

Allow multiple tunnels to the same remote device through different ETH links for
redundancy and load balancing. Requires:
- `set_remote_transfer_ethernet_cores` to accept multiple cores
- Round-robin or least-loaded selection in `get_remote_transfer_ethernet_core()`
- Per-core h2d/d2h tracking (already in place with separate HostToLiteFabricInterface)

Deferred until coexistence is working.

---

## Implementation Order

1. **Phase A+B+C**: Remove ERISC0 kill, remove keepalive, remove defensive re-kill
   (all in lite_fabric FW — single coherent change)
2. **Phase G**: Fix TXQ0 diagnostic to use TXQ2 (part of same FW change)
3. **Phase F**: Remove TXQ0 CTRL write from boot sequence
4. **Phase D**: Ensure `initialize_remote_eth_cores_for_fabric` always sets
   `erisc1_running=true` when lite fabric is active
5. **Phase E**: Verify `update_lite_fabric_bindings_for_fabric_routers` works
   correctly with all cores having fabric routers
6. **Build + test**: Clear FW cache, rebuild, run test script
7. **Phase H+I**: D-cache optimization and multi-path tunnels (future)

## Key Files

| File | Role |
|------|------|
| `tt_metal/lite_fabric/hw/src/lite_fabric.cpp` | FW main loop, ERISC0 kill, keepalive, TXQ init |
| `tt_metal/lite_fabric/hw/inc/channels.hpp` | Sender/receiver logic, sentinel loop keepalive |
| `tt_metal/lite_fabric/hw/inc/constants.hpp` | TXQ assignments, TRID offsets, stream reg IDs |
| `tt_metal/lite_fabric/hw/inc/host_interface.hpp` | FabricLiteConfig, FabricLiteMemoryMap |
| `tt_metal/lite_fabric/hw/inc/init-fsm-basic.hpp` | Init handshake (uses TXQ0 via ConnectedRiscInterface) |
| `tt_metal/lite_fabric/hw/inc/blackhole/risc_interface.hpp` | ConnectedRiscInterface (TXQ0 for remote reg writes) |
| `tt_metal/impl/device/device.cpp` | `configure_fabric()` — ERISC0 deassert on MMIO cores |
| `tt_metal/impl/device/device_manager.cpp` | `init_fabric()` ordering — deep-hop first |
| `tt_metal/impl/context/metal_context.cpp` | `initialize_remote_eth_cores_for_fabric()`, `update_lite_fabric_bindings_for_fabric_routers()` |
| `tt_metal/third_party/umd/.../lite_fabric.hpp` | UMD-side host interface and memory map |
| `tt_metal/third_party/umd/.../remote_communication_lite_fabric.cpp` | UMD read/write/rebind |

## Test Plan
You will be fed the output of this script to debug and complete implementation:
```
import torch
import ttnn

# DeepSeek V3 MLA dimensions
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
MLA_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 (what Q/KV look like after absorb)

MESH_SHAPE = (1, 8)
NUM_HEADS_LOCAL = NUM_HEADS // MESH_SHAPE[1]  # 16
SEQ_LEN = 128


def run():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    with ttnn.create_mesh_device(ttnn.MeshShape(*MESH_SHAPE)) as mesh:
        print(f"Opened mesh: {mesh.shape}")
        grid = mesh.compute_with_storage_grid_size()
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        # Random Q and KV tensors (replicated — no CCLs needed)
        q = ttnn.from_torch(
            torch.randn(1, NUM_HEADS_LOCAL, SEQ_LEN, MLA_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, device=mesh, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT,
        )
        kv = ttnn.from_torch(
            torch.randn(1, 1, SEQ_LEN, MLA_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b, device=mesh, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT,
        )

        scale = QK_HEAD_DIM**-0.5

        out = ttnn.transformer.flash_mla_prefill(
            q, kv,
            head_dim_v=KV_LORA_RANK,
            scale=scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            is_causal=True,
        )

        out_torch = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        print(f"Output shape: {out_torch.shape}  (expect [1, {NUM_HEADS_LOCAL}, {SEQ_LEN}, {KV_LORA_RANK}])")
        print(f"Sample: {out_torch.flatten()[:4].tolist()}")
        print("PASS")


if __name__ == "__main__":
    # Intentionally run twice to verify mesh open/close cycle is repeatable
    run()
    run()
```


Make sure devices 0-7 are fully working for ops and that teardown works successfully
DO NOT BUILD THE CHANGES OR RUN THEM WHEN YOU ARE DONE. I will do this from an extrnel loop.
Start this session by checking your memory for progress and bugs from previous runs.


## Risks

1. **Boot window**: Between Phase 2 (lite fabric starts) and Phase 4 (fabric router
   starts), ERISC0 is in reset. No natural keepalive. BH MAC timeout is ~10s;
   boot window is ~2-5s. Should be fine but monitor.

2. **Init handshake TXQ0**: `ConnectedRiscInterface` uses TXQ0 during Phase 2 init.
   Fabric router hasn't started yet, so no conflict. But if lite fabric re-inits
   after fabric router is running, TXQ0 would conflict. This shouldn't happen in
   normal flow.

3. **L1 overlap**: Fabric router's L1 must not extend into 0x62000-0x70000 (lite
   fabric region). Verify via `MEM_ERISC_MAX_SIZE < 0x62000` (currently ~0x61260).

4. **NOC contention**: Both ERISCs share NOC0. Lite fabric uses per-TRID barriers
   and sentinel-based reads to avoid counter interference. Fabric router also uses
   NOC0 but with different TRIDs (0-7). Verify no overlap.
