# Fabric Router Dispatch for P150 (1x8 Blackhole)

## Current Objective

Get fabric router dispatch working end-to-end for all 8 devices in a 1x8 Blackhole mesh.
Lite fabric bootstrap is DONE and WORKING. The current problem is in the dispatch-through-fabric-router
layer: Device 0 (phys6, 3-hop NORTH from MMIO) stalls during op execution.

## Physical Topology

```
D0(phys6) ── D1(phys4) ── D2(phys2) ── D3(phys0/MMIO) ── D4(phys1) ── D5(phys3) ── D6(phys5) ── D7(phys7)
  3-hop N      2-hop N      1-hop N       (PCIe host)       1-hop S      2-hop S      3-hop S      4-hop S
```

- Board: p150b, 1 PCIe device (phys0 = MMIO), rest are remote via ETH
- Config: FABRIC_1D_RING, MeshShape([1, 8]), 1 routing plane per direction
- UMD tunnels: flat `[mmio, remote]` pairs created by Phase 3c (chip-ID order)

## How Dispatch Through Fabric Works

### Forward Path (Host → Remote Device)
```
Host CQ → PREFETCH_H (MMIO Tensix) → FABRIC_MUX (MMIO Tensix) → Router sender ch0 (MMIO ETH/ERISC0)
  → ETH link → Router receiver (intermediate) → forwarding → Router sender ch1 (intermediate)
  → ... (repeat for each hop) ...
  → Router receiver (destination) → NOC write to PREFETCH_D L1 (remote Tensix)
  → DISPATCH_D → workers
```

### Return Path (Remote Device → Host)
```
DISPATCH_D (remote Tensix) → RETURN_FABRIC_MUX (remote Tensix) → Router sender ch0 (remote ETH/ERISC0)
  → ETH link → Router receiver (intermediate) → forwarding → Router sender ch1 (intermediate)
  → ... (repeat for each hop) ...
  → Router receiver (MMIO) → NOC write to DISPATCH_H L1 (MMIO Tensix)
  → DISPATCH_H → host CQ completion
```

### Key Concepts

- **Routing path (LowLatencyRoutingFields)**: 2-bit fields packed into uint64. Each receiver reads lowest 2 bits, right-shifts by 2. Values: WRITE_ONLY=0b01 (deliver locally), FORWARD_ONLY=0b10 (forward to next hop), WRITE_AND_FORWARD=0b11.
- **For 3-hop route**: `0b01_10_10` = FORWARD, FORWARD, WRITE_ONLY. Each intermediate hop forwards, last hop delivers.
- **`num_hops`**: Computed by `get_num_hops(mmio_id, remote_id)` in relay_mux.cpp. Baked into dispatch kernel binaries as compile-time `NUM_HOPS` macro. Used by CQRelayClient to construct the routing path bitmask via `fabric_set_unicast_route()`.
- **Forwarding connections**: On each intermediate chip, NORTH↔SOUTH router pairs. SOUTH receiver → NORTH sender ch1 (northbound forwarding). NORTH receiver → SOUTH sender ch1 (southbound forwarding).
- **Sender ch0**: Connected to local FABRIC_MUX (worker traffic). **Sender ch1**: Connected to paired router's receiver (forwarded traffic).
- **Routing planes**: Number of ETH links used per direction. Currently 1 (minimum across all chips). Endpoint chips with 1 link in their direction drag the count.

### Dispatch Topology (topology.cpp)

`generate_blackhole_multichip_fabric_1cq_nodes()` creates per-direction relay groups:
- Groups remote devices by `(forwarding_direction, link_index)` key
- NORTH group: D0, D1, D2 share one FABRIC_MUX on MMIO
- SOUTH group: D4, D5, D6, D7 share one FABRIC_MUX on MMIO
- Each remote device gets: PREFETCH_D, DISPATCH_D, DISPATCH_S, RETURN_FABRIC_MUX
- MMIO gets: per-device PREFETCH_H + DISPATCH_H pairs, plus FABRIC_MUX per direction

### MUX → Router Connection

In `fabric.cpp:append_fabric_connection_rt_args()`:
- MUX determines forwarding direction from src/dst fabric node IDs
- Selects ETH channel via `candidate_eth_chans[link_idx]`
- Connects to router's sender ch0 (via SenderWorkerAdapterSpec with NOC coords)
- Two-phase handshake: write location info, then write open_connection_value=1

## Current Bug: D0 (3-hop NORTH) CQ Stall

### Symptom
- D0's CQ completion write pointer stuck at 0x2555a8 (made some initial progress, then stopped)
- Other devices (D1-D7) appear to make progress
- Happens during `flash_mla_prefill` op execution on all 8 devices

### Root Cause Hypothesis: Wrong `num_hops` from UMD Tunnel Fallback

**`get_num_hops(mmio, downstream)`** has 3 resolution tiers:
1. `get_lite_fabric_hop_count(downstream)` — BFS-discovered, correct (returns 3 for phys6)
2. **NEW**: Control plane `get_fabric_route()` — computes actual route length (returns 3)
3. UMD tunnel scan — flat `[mmio, remote]` tunnels, returns `hop=1` for ALL devices (WRONG!)

If tier 1 returns 0 (e.g., lite_fabric_hal_ not available), the old code fell through to tier 3,
giving `hop=1` for phys6. DISPATCH_D would construct routing path `0b01` (WRITE_ONLY at first hop).
Completion packets would be "delivered" at D1 instead of forwarded to MMIO → D0's CQ never advances.

**Fix applied**: Added tier 2 (control plane route length) as fallback before the broken UMD scan.

### Other Possible Causes (if hop count is confirmed correct)
- Forwarding chain congestion under load (all 8 devices using shared links simultaneously)
- Flow control credit leak in forwarding path
- EDM router firmware issue at intermediate hops under sustained traffic
- Router connection handshake failure on D0's RETURN_FABRIC_MUX → SOUTH router

### Diagnostics Added
- `relay_mux.cpp`: Logs which `get_num_hops` tier was used (lite_fabric / control_plane / UMD)
- `dispatch.cpp`: Logs exact `num_hops` for DISPATCH_H (forward) and DISPATCH_D (return)
- `prefetch.cpp`: Logs exact `num_hops` for PREFETCH_H (forward)
- `system_memory_manager.cpp`: At CQ stall (≥10s), dumps all devices' CQ pointers, router connection/flow-control semaphores, EDM status, hop counts
- `device_manager.cpp`: Post-init dumps EDM status, MUX cores, routing info per device, connection semaphores
- `control_plane.cpp`: Pre/post routing plane channel counts per device per direction

## Routing Plane Count (Physical Links)

Currently limited to 1 per direction. The `initialize_dynamic_routing_plane_counts()` function
takes the global minimum across all chips for each direction. Endpoint chips (D0, D7) with only
1 ETH link in their direction drag the count to 1.

The diagnostic logs will show exactly which chip/direction is the bottleneck. If endpoint chips
genuinely only have 1 link, we'd need topology changes (e.g., different MeshShape interpretation)
to use more links on the segments that support it.

## Key Files

| File | Role |
|------|------|
| `tt_metal/impl/dispatch/topology.cpp` | Dispatch node graph, relay group formation |
| `tt_metal/impl/dispatch/kernel_config/relay_mux.cpp` | FABRIC_MUX config, `get_num_hops()` |
| `tt_metal/impl/dispatch/kernel_config/dispatch.cpp` | DISPATCH_H/D config, num_hops usage |
| `tt_metal/impl/dispatch/kernel_config/prefetch.cpp` | PREFETCH_H/D config, num_hops usage |
| `tt_metal/impl/dispatch/system_memory_manager.cpp` | CQ wait + stall diagnostics |
| `tt_metal/impl/device/device_manager.cpp` | Fabric/dispatch init, post-init diagnostics |
| `tt_metal/fabric/fabric.cpp` | `append_fabric_connection_rt_args()` MUX→Router |
| `tt_metal/fabric/control_plane.cpp` | Routing tables, routing planes, `get_fabric_route()` |
| `tt_metal/fabric/compute_mesh_router_builder.cpp` | Forwarding connection pairs |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | EDM router FW |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | Forwarding logic |
| `tt_metal/impl/context/metal_context.cpp` | BFS discovery, lite fabric hop counts |
| `tt_metal/impl/dispatch/kernel_config/fd_kernel.cpp` | `GetUpstreamDeviceId/GetDownstreamDeviceId` |

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
Always look at the previous log before this to make sure there are no regressions.


IMPORTANT:
Always try to make your fixes in as minimal lines of code changed as possible. Debug logs do not count towards this line minimization.
