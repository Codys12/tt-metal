# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CCL helpers for Gemma4 tensor parallelism.

Provides three layers of comms support:

1. Async CCLs — `ccl_allreduce` / `ccl_allgather` use the experimental
   fabric-backed `all_reduce_async` / `all_gather_async`, which overlap
   transfer with compute and avoid blocking the worker grid.

2. Fused matmul+CCL — `ccl_matmul_reduce_scatter_allgather` calls
   `experimental.matmul_reduce_scatter_async` so the matmul output tiles
   stream straight into the reduce-scatter; an `all_gather_async` then
   restores full hidden. Used by row-parallel `o_proj` / `down_proj` to
   replace `linear + all_reduce` with a single fused kernel + AG.

3. Persistent buffer factory — modules call
   `make_reduce_scatter_persistent_buffers` once per (shape, dtype) at
   init time so the fused op has stable scratch space at trace capture.
"""

import ttnn


class CCLManager:
    """Owns the semaphores and CCL sub-device used by the fused/async ops."""

    def __init__(self, mesh_device, num_links=1, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()

        grid = mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )

        # Double-buffered semaphore pools so back-to-back fused ops don't
        # alias the same handle. matmul_reduce_scatter_async needs 3 RS
        # sems per call; all_gather_async needs 2; both want a barrier.
        # all_reduce_async (used in `ccl_allreduce`) wants 2 barriers per
        # call — one for the RS phase and one for the AG phase — so we
        # allocate barriers in pairs and hand them out via either accessor.
        # Without pre-allocated semaphores, all_reduce_async would fall
        # back to ttnn.reduce_scatter / ttnn.all_gather, which call
        # create_global_semaphore at runtime — an enqueue_write_to_core
        # that fatally trips begin_trace_capture.
        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(2)])
            self._barrier_semaphores.append(
                [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(2)]
            )
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

    def get_rs_semaphore(self):
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        sems = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sems[0]

    def get_barrier_semaphores_pair(self):
        """Two barriers from the same pool — for ops that need a distinct
        barrier per phase (e.g. all_reduce_async = RS + AG)."""
        sems = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sems


def _tp_axis(mesh_config):
    return mesh_config.tp_axis


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across the TP axis using the async fabric path.

    Pass pre-allocated barrier / RS / AG semaphores so the C++ side picks
    the ``reduce_scatter_minimal_async`` + ``prim::all_gather_async``
    path. The fall-through (no semaphores) calls ``ttnn.reduce_scatter``
    and ``ttnn.all_gather`` which both run ``create_global_semaphore`` at
    runtime — that allocation enqueues a host write to a core address,
    which is a trace-capture fatal.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

    result = ttnn.experimental.all_reduce_async(
        tensor,
        cluster_axis=_tp_axis(mesh_config),
        mesh_device=ccl_manager.mesh_device,
        barrier_semaphores=ccl_manager.get_barrier_semaphores_pair(),
        rs_global_semaphores=ccl_manager.get_rs_semaphore(),
        ag_global_semaphores=ccl_manager.get_ag_semaphore(),
        math_op=ttnn.ReduceType.Sum,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Linear,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return result


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across the TP axis using the async fabric path."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

    gathered = ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        cluster_axis=_tp_axis(mesh_config),
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        num_links=ccl_manager.num_links,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap; at least 1."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def make_fused_matmul_program_config(
    matmul_output_shape,
    in_dim_per_device,
    core_grid=(8, 6),
    max_in0_block_w=5,
):
    """Build a `MatmulMultiCoreReuseMultiCastProgramConfig` for the row-parallel
    matmul-fused-with-reduce-scatter case.

    Why this is needed: `ttnn.experimental.matmul_reduce_scatter_async` does
    NOT auto-derive `program_config` (unlike regular `ttnn.linear`). Its
    program factory unconditionally calls `program_config.value()` on the
    optional, throwing `bad optional access` when the caller passes None.
    See `matmul_reduce_scatter_async_program_factory.cpp:55`.

    Constraints (TT_FATAL'd at the matmul validate step):
    - `(input_dim_per_device / 32) % in0_block_w == 0` (matmul_device_operation.cpp:1118)
    - `per_core_N % out_block_w == 0`                  (matmul_device_operation.cpp:781)

    matmul_output_shape: full pre-RS shape, e.g. [1, 1, tile_pad, hidden]
    in_dim_per_device:   per-device input dim (e.g. hidden_size // tp for o_proj)
    core_grid:           matmul compute grid; reserve rows beyond this for the
                         reduce-scatter cores via reduce_scatter_core_grid_offset.
    """
    import math

    kt = max(1, in_dim_per_device // 32)
    in0_block_w = _largest_divisor_at_most(kt, max_in0_block_w)
    per_core_M = max(1, math.ceil(matmul_output_shape[2] / 32 / core_grid[1]))
    per_core_N = max(1, math.ceil(matmul_output_shape[3] / 32 / core_grid[0]))
    out_block_w = _largest_divisor_at_most(per_core_N, max(1, per_core_N // 2))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def make_reduce_scatter_persistent_buffers(
    mesh_device,
    matmul_output_shape,
    tp,
    dtype=ttnn.bfloat16,
    memory_config=None,
):
    """Allocate the (intermediate, output) buffers required by
    `matmul_reduce_scatter_async`.

    intermediate: two full matmul outputs, replicated across mesh.  Linear
                  reduce-scatter keeps independent forward/backward partials.
    output:       reduce-scattered output (last dim divided by TP)

    Returns (intermediate_buffer, output_buffer) both ttnn tensors on device.
    Returns (None, None) if TP <= 1.
    """
    if tp <= 1:
        return None, None

    import torch

    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out_shape = list(matmul_output_shape)
    out_shape[-1] = out_shape[-1] // tp

    is_mesh = hasattr(mesh_device, "shape")
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # ccl_matmul_reduce_scatter_allgather uses Topology.Linear.  Its line
    # reduce-scatter implementation addresses two full input-sized scratch
    # regions (one per direction), matching the persistent-buffer convention
    # used by reduce_scatter_minimal_async.
    intermediate_shape = list(matmul_output_shape)
    intermediate_shape[0] *= 2
    intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=mc,
        mesh_mapper=mapper,
    )
    output_buf = ttnn.from_torch(
        torch.zeros(out_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=mc,
        mesh_mapper=mapper,
    )
    return intermediate, output_buf


def ccl_matmul_reduce_scatter_allgather(
    input_tensor,
    weight,
    persistent_intermediate_buffer,
    persistent_output_buffer,
    mesh_config,
    ccl_manager,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    memory_config_mm=None,
    memory_config_rs=None,
    memory_config_ag=None,
    reduce_scatter_core_grid_offset=(0, 0),
):
    """Fused row-parallel matmul + all-reduce.

    Replaces the `linear → all_reduce` pattern used after `o_proj` / `down_proj`
    with `matmul_reduce_scatter_async → all_gather_async`. The matmul output
    tiles are streamed directly into the reduce-scatter, hiding the comm
    behind the matmul. The trailing `all_gather_async` restores full hidden
    so the residual add downstream can run unchanged.

    Falls back to the unfused `linear + ccl_allreduce` path when TP=1 or
    when the persistent buffers are not provided.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        # Don't pass program_config here: it was built for the fused path's
        # matmul validate, which has different constraints than ttnn.linear's.
        # ttnn.linear auto-derives a working config when None.
        out = ttnn.linear(
            input_tensor,
            weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config_mm,
            dtype=dtype,
        )
        input_tensor.deallocate(True)
        return out

    use_fused = persistent_intermediate_buffer is not None and persistent_output_buffer is not None
    if use_fused:
        # Buffer shape must exactly match the matmul output. Otherwise the
        # fused op would over- or under-shoot. Fall back if mismatched.
        in_shape = list(input_tensor.shape)
        w_shape = list(weight.shape)
        expected_seq = in_shape[-2]
        expected_out_dim = w_shape[-1]
        buf_shape = list(persistent_intermediate_buffer.shape)
        if buf_shape[-2] != expected_seq or buf_shape[-1] != expected_out_dim:
            use_fused = False

    if not use_fused:
        # Fallback: same reasoning as TP<=1 — don't pass the fused-tuned
        # program_config to ttnn.linear; it auto-derives.
        out = ttnn.linear(
            input_tensor,
            weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config_mm,
            dtype=dtype,
        )
        input_tensor.deallocate(True)
        return ccl_allreduce(out, mesh_config, ccl_manager, memory_config=memory_config_mm)

    memory_config_mm = memory_config_mm or ttnn.DRAM_MEMORY_CONFIG
    memory_config_rs = memory_config_rs or ttnn.DRAM_MEMORY_CONFIG
    memory_config_ag = memory_config_ag or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = _tp_axis(mesh_config)

    # Both returned tensors alias the persistent buffers — do NOT deallocate
    # them here, the buffers are owned by the calling module.
    _matmul_out, scattered = ttnn.experimental.matmul_reduce_scatter_async(
        input_tensor,
        weight,
        persistent_intermediate_buffer,
        persistent_output_buffer,
        dim=3,
        multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
        reduce_scatter_core_grid_offset=ttnn.CoreCoord(*reduce_scatter_core_grid_offset),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        num_links=ccl_manager.num_links,
        memory_config_rs=memory_config_rs,
        topology=ttnn.Topology.Linear,
        memory_config_mm=memory_config_mm,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    input_tensor.deallocate(True)

    # all_gather_async writes a new tensor; `scattered` (the persistent buffer)
    # remains alive for the next call and is not deallocated here.
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        num_links=ccl_manager.num_links,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config_ag,
    )
    return gathered


def make_block_sharded_matmul_config(m, k_local, n_local, grid=(12, 8), with_out_mem=False):
    """L1-block-sharded matmul config for packed-verify dense matmuls.

    Measured at 512x5376x5376/bf4 on tf1: 0.51 ms (DRAM interleaved auto) →
    0.28 ms (105 TF/dev) — the dense matmul ceiling on Blackhole is mcast
    bandwidth, not FLOPs; block-sharded in0 cuts mcast volume 4×.

    Constraints: in0_block_w must divide K-shard tile count (K/32/gx).
    Returns (program_config, memory_config) — call x.to_memory_config(mem)
    first; output stays block-sharded.
    """
    gx, gy = grid
    if (k_local // 32) % gx:
        gx = 8  # e.g. o_proj K=2048 (64 tiles): 12 cols don't divide; 8 do
    mt, kt, nt = max(1, m // 32), max(1, k_local // 32), max(1, n_local // 32)
    per_core_m = -(-mt // gy)
    per_core_n = -(-nt // gx)
    shard_kt = -(-kt // gx)
    # in0_block_w = full K shard maximizes reuse, but circular buffers are
    # statically allocated: in0/in1 (double-buffered) + output + the L1
    # in0 shard must fit a core's 1.46 MB. Every M=512 shape fits at full
    # shard depth; the 2048 prefill bucket o_proj (K=2048 → 8-col grid,
    # 21 N-tiles/core) overflows — shrink in0_block_w to the largest
    # divisor of the K-shard that fits (kw14 → kw7 costs ~9% at M=512).
    tile_bytes = 2048
    budget = 1_400_000
    if with_out_mem:
        # gate/up chain keeps two L1-sharded outputs resident next to the CBs.
        budget -= 2 * per_core_m * per_core_n * tile_bytes
    kw = shard_kt
    while kw > 1:
        cb_bytes = (
            2 * per_core_m * kw  # in0 CB (double-buffered)
            + 2 * kw * per_core_n  # in1 CB (double-buffered)
            + 2 * per_core_m * per_core_n  # out CB + partials
            + per_core_m * shard_kt  # in0 L1 shard
        ) * tile_bytes
        if cb_bytes <= budget:
            break
        kw = max((d for d in range(1, kw) if shard_kt % d == 0), default=1)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kw,
        out_subblock_h=1,
        out_subblock_w=max(d for d in (8, 7, 4, 2, 1) if per_core_n % d == 0),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        out_block_w=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    mem = ttnn.create_sharded_memory_config(
        (per_core_m * 32, shard_kt * 32),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if not with_out_mem:
        return pc, mem
    # L1 block-sharded output: saves the DRAM write when the consumer is the
    # next matmul on the same grid (gate/up → mul → down_proj).
    out_mem = ttnn.create_sharded_memory_config(
        (per_core_m * 32, per_core_n * 32),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return pc, mem, out_mem
