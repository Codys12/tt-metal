# Full ERISC0 + ERISC1 Coexistence on All Connected ETH Tiles

## Goal

Every connected ETH core on every chip (MMIO and remote) simultaneously runs:
- **ERISC0**: Metal's fabric router (for op-level CCL communication)
- **ERISC1**: Lite fabric relay (for UMD-level L1 reads/writes to remote devices)

They coexist on the same ETH tile because they use separate L1 regions, separate NOC
TRIDs, and separate TXQs (fabric router: TXQ0/TXQ1, lite fabric: TXQ2).

## Completed Work

### Dual-channel architecture (Groups 1-5)
- **Ch0**: outbound commands (host→remote writes + read commands)
- **Ch1**: inbound read responses (remote→host)
- 4 buffer slots per channel for pipelining
- Per-TRID NOC barriers: ch0 uses TRIDs 8-11, ch1 uses TRIDs 12-15
- Stream register assignments: ch0 IDs 23-25, ch1 IDs 26-28
- UMD: `HostToLiteFabricInterface` has `recv_ch1` state, `flush_recv_ch1_h2d()`,
  `read_one_page()` uses ch1 receiver buffers
- Memory layout: 56KB (base 0x62000), FW and UMD memory maps in sync

### Lite fabric / fabric router resource separation
- Lite fabric: TXQ2, TRIDs 8-15, stream regs 23-28, L1 region 0x62000-0x70000
- Fabric router: TXQ0/TXQ1, TRIDs 0-7, stream regs 0-22+29-31, L1 below 0x62000

### N-hop BFS discovery (Phase 2b)
- BFS discovers multi-hop chips, launches downstream tunnels, configures forwarding
- Forwarding stays on ch0 only; direct 1-hop reads use ch1
- `downstream_sender_cores_` tracks which cores have ERISC1 running as downstream senders

### Phase A: ERISC0 kill + TXQ0 init (DONE)
**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

- ERISC0 killed at boot via local RISC-V store (0x46800). Required because init-fsm
  uses TXQ0, ERISC0 syseng FW also uses TXQ0 → contention causes hardware hang.
- TXQ0 KEEPALIVE enabled after kill. After init, ERISC1 switches to TXQ2 steady-state.
- Periodic keepalive on TXQ2 every ~65K iterations (safety net for Phase 2→4 window).
- Keepalive in sentinel spin loop (channels.hpp NOC_READ) prevents ETH timeout during slow reads.

### Phase B: L1-based notification counters (DONE)
**Files: `channels.hpp`, `lite_fabric.cpp`**

- COMMAND frames (stream register updates) are TXQ0-only on BH; lite fabric uses TXQ2.
- Replaced with L1 DATA frame counters: `pkts_sent_notify[2][4]`, `pkts_completed_notify[2][4]`.
- Read via `noc_self_read_word()` to bypass D-cache (BSS variables have stale cache lines).

### Phase C: NOC cmd buffer separation (DONE)
**Files: `channels.hpp`, `constants.hpp`, `lite_fabric.cpp`**

- ERISC1: cmd buf 2 (write), cmd buf 3 (read). ERISC0: cmd buf 0/1.
- `local_chip_data_cmd_buf = DYNAMIC_NOC_NCRISC_WR_CMD_BUF`, `LF_RD_CMD_BUF = 3`.
- Initialized in main() mirroring noc_init's setup.

### Phase D: TXQ2 MAC configuration (DONE)
**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

- TXPKT_CFG_SEL_SW (0x80) and TXPKT_CFG_SEL_HW (0x84) set to entry 0 (broadcast MAC DA).
- Without this, TXQ2 uses undefined config → remote MAC drops frames silently.

### Phase E: Init-fsm TXQ0 serialization (DONE)
**File: `tt_metal/lite_fabric/hw/inc/init-fsm-basic.hpp`**

- `k_DataTxq = 0`: data + handshake on same TXQ as COMMAND frames (deassert).
- Natural serialization, no cross-TXQ race. TXQ busy barrier after binary send.

### Phase F: Per-core erisc1_running detection (DONE)
**File: `tt_metal/impl/context/metal_context.cpp`**

- erisc1_running=true only for: MMIO-peering, downstream senders, tunnel endpoints.
- Non-tunnel cores keep ERISC1 in POR (0x47000). Prevents syseng subordinate FW conflicts.

### Phase G: Early MetalContext::initialize (DONE)
**File: `tt_metal/distributed/mesh_device.cpp`**

- MeshDevice::create calls initialize() before control plane access.
- Ensures BFS discovery completes before SystemMesh construction.

### device.cpp configure_fabric (DONE — no changes needed)
- SOFT_RESET_BOTH_RUNNING = 0x46000. TXQ partitioning eliminates contention.

---

## Remaining Work

### Build + Test (NEXT)

1. Clear FW cache: `rm -rf ~/.cache/tt-metal-cache/`
2. Build: `cmake --build build -- -j$(nproc) tt_metal`
3. Run test script (see Test Plan below)
4. Debug any failures

### Phase H: D-Cache Optimization (Nice-to-Have, DEFERRED)

**File: `tt_metal/lite_fabric/hw/src/lite_fabric.cpp`**

The `noc_self_read_word()` calls are slow (NOC DMA self-read to bypass D-cache).
Note: `invalidate_l1_cache()` on BH is just `asm("fence")` — it does NOT invalidate
D-cache lines. CSR 0x7c0 bit 3 prevents NEW allocations but existing stale lines
from `data_init()` (which runs before CSR set on second boot) persist.

Deferred until coexistence is working.

### Phase I: Multi-Path Tunnels (Nice-to-Have, DEFERRED)

Allow multiple tunnels to the same remote device through different ETH links for
redundancy and load balancing. Requires:
- `set_remote_transfer_ethernet_cores` to accept multiple cores
- Round-robin or least-loaded selection in `get_remote_transfer_ethernet_core()`
- Per-core h2d/d2h tracking (already in place with separate HostToLiteFabricInterface)

Deferred until coexistence is working.

---

## 6-Phase Initialization Order

1. **Phase 1**: MMIO FW — launch Metal FW on MMIO device ETH cores (ERISC0)
2. **Phase 2**: Lite fabric — launch ERISC1 on all connected MMIO ETH cores, init-fsm handshake with neighbors
3. **Phase 2a**: Upgrade remote SoC — read boot_results, populate SoC descriptor
4. **Phase 2b**: BFS + downstream tunnels — discover N-hop chips, launch downstream senders/receivers
5. **Phase 3**: Remote FW — launch ERISC0 (Metal active erisc) on all remote device ETH cores via `initialize_remote_eth_cores_for_fabric()`
6. **Phase 4**: Fabric router — `configure_fabric()` on all devices (deep-hop first, 1-hop second, MMIO last)

After Phase 4, every connected ETH core has ERISC0 (fabric router) + ERISC1 (lite fabric relay) running simultaneously.

## Key Files

| File | Role |
|------|------|
| `tt_metal/lite_fabric/hw/src/lite_fabric.cpp` | FW main loop — ERISC1 only touches TXQ2 |
| `tt_metal/lite_fabric/hw/inc/channels.hpp` | Sender/receiver logic, TXQ2 recovery |
| `tt_metal/lite_fabric/hw/inc/constants.hpp` | TXQ assignments (DEFAULT_ETH_TXQ=2), TRID offsets, stream reg IDs |
| `tt_metal/lite_fabric/hw/inc/host_interface.hpp` | FabricLiteConfig, FabricLiteMemoryMap |
| `tt_metal/lite_fabric/hw/inc/init-fsm-basic.hpp` | Init handshake (uses TXQ0 via ConnectedRiscInterface — Phase 2 only) |
| `tt_metal/lite_fabric/hw/inc/blackhole/risc_interface.hpp` | ConnectedRiscInterface (TXQ0 for remote reg writes — Phase 2 only) |
| `tt_metal/impl/device/device.cpp` | `configure_fabric()` — ERISC0 deassert with 0x46000 on all cores |
| `tt_metal/impl/device/device_manager.cpp` | `init_fabric()` ordering — deep-hop first, 1-hop second, MMIO last |
| `tt_metal/impl/context/metal_context.cpp` | `initialize_remote_eth_cores_for_fabric()` (erisc1_running=true when lite fabric active), `update_lite_fabric_bindings_for_fabric_routers()` |
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


## Risks and Known Caveats

1. **Boot window (Phase 2 → Phase 4)**: Between lite fabric start and fabric router
   start, ERISC0 is in POR/reset state. No natural keepalive from ERISC0. BH MAC
   timeout is ~10s; boot window is ~2-5s. BFS traffic provides some ETH activity.
   If link timeouts are observed during boot, add a conditional keepalive in ERISC1
   that checks whether ERISC0 is running (e.g., read TXQ0 CTRL or a shared flag).

2. **Init handshake TXQ0**: `ConnectedRiscInterface` uses TXQ0 during Phase 2 init
   (before fabric router starts in Phase 4). No conflict in normal flow. Would
   conflict if lite fabric re-initializes after fabric router is running — this
   doesn't happen in normal operation. Low-priority future fix: migrate
   ConnectedRiscInterface to TXQ2.

3. **L1 overlap**: Fabric router's L1 must not extend into 0x62000-0x70000 (lite
   fabric region). Currently safe: `MEM_ERISC_MAX_SIZE` is ~0x61260. Monitor if
   fabric router grows.

4. **NOC contention**: Both ERISCs share NOC0. Lite fabric uses per-TRID barriers
   (TRIDs 8-15) and sentinel-based reads to avoid counter interference with
   fabric router (TRIDs 0-7). No overlap verified.

5. **ERISC0 killed at boot**: ERISC1 kills ERISC0 at boot (local store to 0x46800)
   to prevent TXQ0 contention during init-fsm. ERISC0 remains in reset until
   Phase 3 (remote FW) or Phase 4 (configure_fabric on MMIO) relaunches it.
   During Phase 2→4 window, no fabric router traffic exists — the periodic
   keepalive on TXQ2 prevents ETH link timeout.


IMPORTANT:
You need to make sure lite_fabric deployment is working first across at least 8 total devices (there are at least 7 remote for you to used cabled up)
Then you need to make sure fabric router is working across those ETH tiles for all ERISC0s in tandem with the lite fabric logic.

Use for context the commit "working!" for single ttnn.open_device bringup. That successfully loaded lite_fabric across all remote chips and should be used as a baseline/reference for debugging why lite fabric is not working for n-hop. Study it carefully when you need to. Lite fabric should be fully deployed before you go on to fabric router for maximum simplicity. You must be very attentive to the lite fabric setup/bringup across n-hop devices, as even the slightest mistake can lead to a hang. This lite fabric will be a persistant control plane once set up -- ideally all through TXQ2.

YOU REALLY DO WANT TXQ2 AFTER YOU REACH STEADY STATE (AFTER ALL LITE FABRIC IS SET UP BUT BEFORE FABRIC ROUTER IS LAUNCHED)
Finally: always try to make your fixes in as minimal lines of code changed as possible. Debug logs do not count towards this line minimization.
