# N-Hop Lite Fabric: Comprehensive Implementation Plan

## Goal

Enable lite fabric to bootstrap remote devices that are N hops away from any MMIO device, using iterative discovery (BFS). The topology is **not** known ahead of time — we probe chip-by-chip to build the full cluster graph. The same lite fabric firmware binary gains a forwarding code path so intermediate hops relay packets to further devices.

Remote chip <-> remote chip links need to be included in the topology mapping .

Some important context: The machine you are on has 8 cards in a ring, with one dangling card you should ignore. Only one card is on PCIe. You need to discover all 8, as device 7 is used in the demo.

---

## Current State Summary

### What exists today (1-hop only)

1. **UMD topology discovery** (`topology_discovery.cpp`): BFS loop discovers MMIO chips via PCIe, then scans their ETH cores to find 1-hop remote chips. Remote BH chips are added to the cluster descriptor but **skipped for ETH scanning** (line 154: `!chip->is_mmio_capable()` → `continue`).

2. **Lite fabric firmware** (`lite_fabric.cpp`): Single binary runs on ERISC1 of both MMIO and remote ETH cores. MMIO side is "sender", remote side is "receiver". Main loop calls `run_sender_channel_step<0>()` + `run_receiver_channel_step<0>()`.

3. **Packet header** (`header.hpp`): `FabricLiteHeader` already has `LiteFabricRoutingFields routing_fields` with 2-bit-per-hop encoding: `NOOP(00)`, `WRITE_ONLY(01)`, `FORWARD_ONLY(10)`, `WRITE_AND_FORWARD(11)`. The `to_chip_unicast(distance_in_hops)` method already generates correct multi-hop routing fields (FORWARD_ONLY for intermediate hops, WRITE_ONLY for final hop). **This is already implemented but unused.**

4. **UMD packet formatting** (`lite_fabric.hpp`): Hardcodes `header.to_chip_unicast(1)` everywhere. Needs parameterization.

5. **Init FSM** (`init-fsm-basic.hpp`): `routing_init()` handles the MMIO→remote handshake. The MMIO primary copies its binary + config to the connected remote core via `eth_send_packet`, then handshakes. Only supports tunnel depth 1.

6. **Host-side launch** (`blackhole_impl.cpp`): `BlackholeLiteFabricHal::launch()` iterates `tunnels_from_mmio`, writes config + binary to each MMIO ETH core, deasserts ERISC1, waits for READY state.

7. **`TunnelDescriptor`**: Stores `mmio_id`, `connected_id`, `num_hops=1`. Only describes direct MMIO↔remote links.

8. **metal_context.cpp phases**: Phase 1 (MMIO FW) → Phase 2 (lite fabric launch + UMD binding) → Phase 2b (remote chip info upgrade) → Phase 3 (remote device build/init/FW launch) → fabric router init → dispatch.

---

## Architecture Decision: Extend Lite Fabric In-Place

**One firmware binary.** No separate relay firmware. The existing `lite_fabric.cpp` gains a forwarding code path activated by the routing fields already present in the packet header.

### Key Principle

Every ERISC1 running lite fabric behaves the same way. The difference between "endpoint" and "relay" is purely config-driven: whether it has a downstream ETH link to forward to.

---

## Implementation Plan

### Phase 1: Firmware Forwarding Path

**Files**: `channels.hpp`, `lite_fabric.cpp`, `host_interface.hpp`

#### 1.1 Add forwarding config to `FabricLiteConfig`

Add a small forwarding table to `FabricLiteConfig` in `host_interface.hpp`:

```cpp
struct ForwardingConfig {
    uint8_t enabled;           // 0 = endpoint only, 1 = relay mode
    uint8_t downstream_txq;    // ETH TXQ to use for forwarding (always 0 for now)
    uint8_t reserved[14];      // Pad to 16B alignment
};
```

Add `ForwardingConfig forwarding` to `FabricLiteConfig`. This is written by the host during the discovery loop when configuring an intermediate hop.

**Risk**: Must not exceed `LITE_FABRIC_CONFIG_SIZE` (9KB). Current `FabricLiteMemoryMap` fits; 16 bytes is safe.

#### 1.2 Add forwarding logic to receiver channel

In `channels.hpp`, modify `service_fabric_request()`:

Before processing the packet, inspect `routing_fields`:

```cpp
// Extract the current hop's 2-bit routing field (lowest 2 bits)
uint32_t current_hop_action = header.routing_fields.value & LiteFabricRoutingFields::FIELD_MASK;

if (current_hop_action == LiteFabricRoutingFields::FORWARD_ONLY ||
    current_hop_action == LiteFabricRoutingFields::WRITE_AND_FORWARD) {
    // Shift routing fields right by FIELD_WIDTH to consume this hop
    // Write shifted header to sender buffer for forwarding
    // Trigger sender channel to forward downstream
}

if (current_hop_action == LiteFabricRoutingFields::WRITE_ONLY ||
    current_hop_action == LiteFabricRoutingFields::WRITE_AND_FORWARD) {
    // Existing local write/read/write_reg logic (unchanged)
}

if (current_hop_action == LiteFabricRoutingFields::FORWARD_ONLY) {
    // Skip local write — just forward
    return;
}
```

**Implementation detail**: Forwarding reuses the sender channel. The receiver copies the packet (with shifted routing fields) into the sender buffer slot, increments `h2d.sender_host_write_index`, and the normal sender channel step forwards it via `eth_send_packet_bytes_unsafe()` to the next hop's receiver buffer.

This works because **at a relay node, the host never writes to the sender channel** — the sender channel is exclusively used for forwarding responses and forwarded packets. The host only writes to the MMIO-side sender channel (hop 0).

**For reads**: The response packet travels back. The relay's receiver (on the return path) sees a packet from the downstream direction and forwards it upstream to the MMIO sender. This requires the relay to accept packets from both directions — but this is already the case: the existing receiver channel receives from the ETH link, and the existing sender channel sends to the ETH link. The directions are symmetric. We may need a second receiver/sender channel pair for the return path, or we can reuse the single channel with careful ordering (since the credit system already prevents buffer overflow).

**Simplest approach for reads**: Initially, **do not support reads through multi-hop**. Reads are only used during `upgrade_remote_bh_chip_info` (Phase 2b) to read harvesting info. For n-hop devices, we can defer this or use WRITE_REG-based alternatives. This drastically simplifies the initial implementation.

#### 1.3 Modify `service_lite_fabric()` main loop

No changes needed — `run_sender_channel_step<0>()` already picks up packets in the sender buffer regardless of who put them there (host or receiver forwarding logic). The self-healing check for `num_free_slots` may need adjustment if the receiver is producing packets into the sender buffer.

---

### Phase 2: Host-Side Discovery Loop

**Files**: `metal_context.cpp`, `lite_fabric_hal.cpp`, `lite_fabric_hal.hpp`

#### 2.1 Replace flat `tunnels_from_mmio` with a tree structure

Replace the flat vector with a tree that tracks hop-by-hop paths:

```cpp
struct TunnelDescriptor {
    ChipId mmio_id;                    // Root MMIO chip
    CoreCoord mmio_core_virtual;       // MMIO ETH core (first hop sender)
    CoreCoord mmio_core_logical;
    ChipId connected_id;               // Final destination chip
    CoreCoord connected_core_virtual;
    CoreCoord connected_core_logical;
    int num_hops;                      // Total hops from MMIO
    // NEW: intermediate hop chain for multi-hop
    struct HopInfo {
        ChipId chip_id;
        CoreCoord eth_core_logical;     // ETH core on this chip used for relay
        CoreCoord eth_core_virtual;
    };
    std::vector<HopInfo> intermediate_hops;  // Empty for 1-hop, populated for n-hop
};
```

#### 2.2 Iterative BFS discovery in `metal_context.cpp`

Replace the current linear flow with a BFS loop:

```
discovered = {MMIO chips}  // Already booted in Phase 1
frontier = {MMIO chips}    // Chips whose ETH links we need to probe

while frontier is not empty:
    next_frontier = {}

    for each chip C in frontier:
        for each active ETH core on C:
            if ETH link is trained AND neighbor is not in discovered:
                new_chip = neighbor chip ID

                // 1. The 2-erisc dance already happened when C's ERISC0 booted
                //    (for MMIO chips, during Phase 1; for remote chips, when we
                //    deasserted their ERISC0 in a previous iteration).
                //    So new_chip's ERISC1 receiver is already alive.

                // 2. If C is MMIO: lite fabric sender is already running on C's ERISC1
                //    If C is remote: configure C's ERISC1 forwarding to new_chip
                //    (write ForwardingConfig via existing lite fabric path)

                // 3. We can now reach new_chip via lite fabric chain:
                //    MMIO → ... → C → new_chip

                // 4. Add tunnel descriptor with full hop chain
                // 5. Bind UMD for new_chip
                // 6. Read new_chip's harvesting info (via lite fabric writes/WRITE_REG)
                // 7. Build and init new_chip (write Tensix FW, deassert Tensix cores)
                // 8. Boot new_chip's ERISC0 (write FW, trampoline, deassert)
                //    → This triggers 2-erisc dance with new_chip's further neighbors
                //    → Those neighbors' ERISC1 receivers come alive

                discovered.add(new_chip)
                next_frontier.add(new_chip)

    frontier = next_frontier

// Now all devices are discovered and booted
// Compute control plane routing tables from full topology
// Deploy routing tables to all devices via lite fabric chain
// Launch fabric routers
// Set up dispatch and command queues
```

#### 2.3 ETH link probing for remote chips

Currently, remote BH chips are skipped during UMD topology discovery because lite fabric isn't running yet. For the BFS loop, we need to probe a remote chip's ETH links **after** lite fabric reaches it.

Three approaches (in order of simplicity):

**Option A: Host-side probing via lite fabric reads**
- Read `port_status` (0x7CC04) on each ETH core of the newly discovered chip via lite fabric L1 read
- Read `remote_asic_id` (0x7CFE1 etc.) for trained links
- This uses existing lite fabric read path — no firmware changes needed
- **Downside**: Requires multi-hop reads to work, which we may not initially support

**Option B: Firmware-assisted probing**
- After booting a remote chip's ERISC0, have it write its ETH link status to a known L1 location
- Host reads that L1 location via lite fabric read
- Could be done as part of the active_erisc.cc boot sequence: ERISC0 probes all ETH cores and writes a link status bitmap to a fixed L1 address

**Option C: Use 1-hop reads from the most recently booted chip (recommended)**
- After booting a remote chip and establishing a lite fabric tunnel to it, read its ETH link status using 1-hop lite fabric reads from its directly connected parent
- No multi-hop reads needed — we always read from the most recently booted chip, which is 1 hop from its parent relay (or from the MMIO chip for the first hop)
- Each newly booted ERISC0 writes `{port_status, remote_asic_id}` for all 12 ETH cores to a status region in L1 during boot, OR we just do direct 1-hop reads to the syseng boot results addresses (0x7CC04, 0x7CFE1, etc.) since these are populated by the base ERISC firmware that ran at POR

**Recommendation**: Option C. We always have a direct 1-hop lite fabric tunnel to the chip we just booted (its parent is either MMIO or a relay we set up in the previous BFS iteration). Reads from 1 hop away already work. No multi-hop read support needed for discovery.

**Critical detail**: The syseng base firmware boot results (port_status at 0x7CC04, remote_asic_id at 0x7CFE1/0x7CFF4, etc.) are written by the ERISC syseng FW that runs at power-on reset, **before** Metal even starts. These values are already present in L1 on all ETH cores (including those on remote chips). We just need a lite fabric read path to access them.

---

### Phase 3: UMD Changes

**Files**: `lite_fabric.hpp` (UMD), `remote_communication_lite_fabric.cpp`, `topology_discovery.cpp`

#### 3.1 Parameterize hop count in UMD packet formatting

Change all `header.to_chip_unicast(1)` to `header.to_chip_unicast(num_hops)` where `num_hops` comes from the tunnel descriptor.

In `lite_fabric.hpp`:
```cpp
void write(..., uint8_t distance_in_hops = 1) {
    header.to_chip_unicast(distance_in_hops);
    ...
}
```

The `HostToLiteFabricInterface` needs to know the hop count for the tunnel it's associated with. Add `uint8_t num_hops` to the interface state.

#### 3.2 Multi-hop UMD routing

`set_remote_transfer_ethernet_cores()` already binds a remote chip to specific ETH channels on the gateway MMIO device. For n-hop, the gateway is always the MMIO chip — the hop chain is transparent. UMD writes to the MMIO ETH core's sender buffer; the routing fields in the packet handle the rest.

No fundamental change to UMD routing — just the hop count in the header.

#### 3.3 Deferred remote chip ETH scanning

In `topology_discovery.cpp`, the BH remote chip skip (line 154) stays as-is for the initial UMD discovery. The BFS loop in `metal_context.cpp` handles incremental discovery after lite fabric is running. New chips discovered during the BFS need to be added to the cluster descriptor dynamically:

- Add a method to `ClusterDescriptor` to register new chips and connections after initial construction
- Or: restructure so UMD only discovers MMIO chips, and Metal's BFS loop builds the full cluster descriptor

**Recommendation**: Keep UMD discovery as MMIO-only for BH. Metal owns the full topology discovery via the BFS loop after lite fabric is available. This is the least disruptive change to UMD.

---

### Phase 4: Init Sequence Restructuring

**Files**: `metal_context.cpp`, `device_manager.cpp`

The current Phase 1→2→2b→3 sequence becomes iterative:

```
Phase 1: Boot MMIO devices (unchanged)

Phase 2: BFS Discovery + Lite Fabric Extension
    Launch 1-hop lite fabric (existing code)
    BFS loop:
        For each newly reachable chip:
            a. Probe its ETH links (read link status from parent relay)
            b. Build TunnelDescriptor with full hop chain
            c. Bind UMD for this chip
            d. Read harvesting info (upgrade_remote_bh_chip_info)
            e. build_and_init_devices (Tensix FW)
            f. Boot ERISC0 (FW + trampoline + deassert)
               → 2-erisc dance brings up next-hop ERISC1 receivers
            g. Configure forwarding on this chip's ERISC1
               (write ForwardingConfig via lite fabric)
            h. Add newly discovered neighbors to BFS frontier

Phase 3: Control plane + fabric routers (unchanged, but now has full topology)

Phase 4: Dispatch + command queues (unchanged)
```

---

### Phase 5: Completion/ACK Propagation for Multi-Hop

For writes, the completion path is:
1. Final-hop receiver completes NOC write, sends completion to its sender (upstream)
2. Upstream relay's receiver gets the completion, forwards it further upstream
3. Eventually reaches MMIO sender, which updates `d2h.fabric_sender_channel_index`

**This already works with the forwarding logic in Phase 1** — completions are stream register updates via `remote_update_ptr_val`, which travel as ETH register writes. Each hop's completion is independent: the MMIO sender waits for its direct receiver (1-hop relay) to ack, the 1-hop relay waits for the 2-hop receiver, etc. The credit system at each hop prevents buffer overflow.

**Important nuance**: The MMIO sender's completion means "the packet has been forwarded by the 1-hop relay", NOT "the packet has been written to the final destination". For correctness, `l1_barrier()` needs end-to-end semantics. Options:

- **Option A**: Each relay only sends completion upstream after receiving completion from downstream. This provides end-to-end guarantees but adds latency per hop.
- **Option B**: Keep per-hop completion and add an explicit end-to-end barrier using a special WRITE_REG or NOC read from the final destination.

**Recommendation**: Option A for correctness. The relay's `run_receiver_channel_step` should only increment `completion_counter` (and thus send upstream ack) when the forwarded packet's downstream ack has been received. This is a natural extension of the existing `transaction_flushed()` check — instead of checking the local NOC write trid, the relay checks the downstream sender's completion counter.

---

## Implementation Order (Lowest Risk First)

### Step 1: Firmware forwarding (Phase 1)
- Add `ForwardingConfig` to `FabricLiteConfig`
- Add routing field inspection + forwarding in `service_fabric_request()`
- Test with simulated 2-hop setup (manually configure relay on existing 1-hop)
- **No changes to discovery or UMD yet** — test by manually setting up forwarding config from host

### Step 2: UMD hop count parameterization (Phase 3.1)
- Change `to_chip_unicast(1)` → `to_chip_unicast(num_hops)`
- Add `num_hops` to tunnel/interface state
- Test end-to-end with manually configured 2-hop

### Step 3: ETH link probing from remote chips (Phase 2.3)
- Add ERISC0 boot-time link status reporting to `active_erisc.cc`
- Test reading link status from 1-hop remote devices

### Step 4: BFS discovery loop (Phase 2.2)
- Restructure `metal_context.cpp` init sequence
- Implement iterative discovery and lite fabric extension
- Test with actual n-hop topology

### Step 5: End-to-end completion semantics (Phase 5)
- Modify relay completion to wait for downstream ack
- Verify `l1_barrier()` provides end-to-end guarantees

### Step 6: Control plane + dispatch for n-hop devices (Phase 4)
- Verify control plane routing works with dynamically discovered topology
- Verify dispatch topology assigns single-chip dispatch to n-hop devices
- End-to-end test: run ops on n-hop remote devices

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Relay buffer exhaustion (store-and-forward at each hop) | Packets dropped or deadlock | Credit system already prevents overflow; 2 buffers per hop is tight but sufficient for sequential ops |
| ERISC1 L1 space for forwarding | Binary too large | Forwarding is ~50 lines of code in existing binary; no new buffers needed (reuse sender buffer) |
| Latency increase per hop | Slower remote ops | Acceptable for bootstrapping; dispatch is local once CQ is set up |
| NOC counter mismatch at relay | Hang | Already solved by `ncrisc_noc_counters_init()` fix; relay doesn't have this issue since forwarding uses ETH send, not NOC |
| ETH link down during chain | Partial cluster | Existing tunnel failure handling (remove failed tunnel, proceed with remaining). BFS naturally handles partial connectivity |
| Multi-hop reads | Complex response routing | Defer reads — use 1-hop reads from parent relay for discovery. Full multi-hop reads can be added later |
| `FabricLiteConfig` size overflow | FW crash | ForwardingConfig is 16 bytes; well within margin |
| Discovery loop never terminates | Hang at init | BFS terminates naturally; add max-hop-count safety limit (e.g., 16 hops) |

---

## What Does NOT Change

- Lite fabric binary is still one binary for all ERISC1 cores
- Lite fabric is still JIT-compiled at runtime (same build flow)
- Dispatch topology for remote devices is still single-chip (no MUX/DEMUX)
- Fabric router on ERISC0 is unchanged
- The 2-erisc dance mechanism is unchanged
- UMD's `RemoteCommunicationLiteFabric` class API is unchanged (just parameterize hop count internally)
- Command queue initialization for remote devices is unchanged (writes go through lite fabric chain)


# Test loop
You will be fed the output of this script to debug and complete implementation:
```
import torch
import ttnn

DEVICE_IDS = [0, 1, 3, 5, 7]


def run_core_logic(device_id: int) -> None:
    device = ttnn.open_device(device_id=device_id)
    try:
        torch_input_tensor_a = torch.rand(4, 7, dtype=torch.float32)
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.exp(input_tensor_a)
        torch_output_tensor = ttnn.to_torch(output_tensor)

        torch_input_tensor_b = torch.rand(7, 1, dtype=torch.float32)
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        matmul_output_tensor = input_tensor_a @ input_tensor_b
        torch_matmul_output_tensor = ttnn.to_torch(matmul_output_tensor)

        print(f"device {device_id} matmul output:")
        print(torch_matmul_output_tensor)
    finally:
        ttnn.close_device(device)


for device_id in DEVICE_IDS:
    run_core_logic(device_id)
```

Use it to determine what is going wrong with the multi hop device
For reference, the on hop remote devices were working and running ops at the last commit 'working!' Use that as your stable baseline

DO NOT BUILD THE CHANGES OR RUN THEM WHEN YOU ARE DONE. I will do this from an extrnel loop.
Start this session by checking your memory for progress and bugs from previous runs.
