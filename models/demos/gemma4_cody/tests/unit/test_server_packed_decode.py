# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer packed-decode pytest.

Loads ONE Gemma4 decoder layer (no embedding, no lm_head) and runs one
packed multi-token decode forward through it. Selectable between
``sliding_attention`` and ``full_attention`` layer types — the two
arms exercise different RoPE / KV-cache / SDPA paths inside the
attention sub-module.

Production-matching constants:
  * batch        = 32                         (DECODE_BATCH)
  * packed P     = 4 tokens / user            (= T+1 with T=3 drafts)
  * block_size   = 64
  * sliding cache = 1024
  * max_user_seq = 8192                       (test-only override)
  * PV S_k cap   = min(_PV_SK_CAP, 8192) snapped to a block multiple
  * KV cache     = bfloat16
  * fused reduce-scatter disabled             (server forces unfused o_proj path)

The forward is invoked directly — no ``begin_trace_capture`` / ``execute_trace``.

Run (pick one layer type):
    cd /mnt/nas/scratch && source ./python_env/bin/activate
    export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch \\
           HF_MODEL=/mnt/nas/gemma TT_CACHE_PATH=/mnt/nas/gemma_cache \\
           MESH_DEVICE=8xP150
    pytest -s models/demos/gemma4_cody/tests/unit/test_server_packed_decode.py -k "1x4 and sliding"
    pytest -s models/demos/gemma4_cody/tests/unit/test_server_packed_decode.py -k "1x4 and global"
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_cody.config import MeshConfig, ModeConfig
from models.demos.gemma4_cody.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4_cody.tt.ccl import CCLManager
from models.demos.gemma4_cody.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4_cody.tt.model import create_rope_caches
from models.demos.gemma4_cody.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import parametrize_mesh_with_fabric

# Production constants (verbatim from server.py).
DECODE_BATCH = 32
BLOCK_SIZE = 64
SLIDING_CACHE_LEN = 1024
MAX_USER_SEQ_LEN = 8192
PV_SK_CAP_REQUEST = 4096
NUM_DRAFTS = 3  # T=3 → P=4 packed tokens / user

# layer_type → layer_idx in the default ["sliding"]*5 + ["full"] pattern.
_LAYER_IDX_BY_TYPE = {"sliding": 0, "global": 5}


def _alloc(mesh_device, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
    return ttnn.from_torch(t, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=mapper)


def _host(mesh_device, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
    return ttnn.from_torch(t, layout=layout, dtype=dtype, mesh_mapper=mapper)


def _disable_fused_reduce_scatter(layer):
    """Force the unfused o_proj / down_proj path that the server runs at batch=32."""
    for module in (layer.self_attn, getattr(layer, "shared_mlp", None)):
        if module is None:
            continue
        for attr in ("_fused_intermediate", "_fused_output"):
            buf = getattr(module, attr, None)
            if buf is not None:
                try:
                    buf.deallocate(True)
                except Exception:
                    pass
                setattr(module, attr, None)


# Production runs each replica as a 1x4 TP submesh (a 1x8 parent split into two
# data-parallel halves). Pin to (1,4) so the test profiles the exact production
# shape in a single pass instead of fanning out over (1,1)/(1,2)/(1,4)/(1,8).
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("layer_type", ["sliding", "global"], ids=["sliding", "global"])
@pytest.mark.timeout(86400)
def test_single_layer_packed_decode(mesh_device, layer_type):
    """One Gemma4 decoder layer, packed-decode P=4 forward, no trace.

    No embedding, no lm_head — hidden_states is generated directly as a random
    bf16 tensor in the shape the layer expects post-embedding.
    """
    torch.manual_seed(0)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    B = DECODE_BATCH
    P = NUM_DRAFTS + 1
    layer_idx = _LAYER_IDX_BY_TYPE[layer_type]
    is_sliding = layer_type == "sliding"

    # Page-pool layout — bit-identical to Engine.__init__.
    blocks_per_user_max = MAX_USER_SEQ_LEN // BLOCK_SIZE  # 128
    sliding_cache_len = min(SLIDING_CACHE_LEN, MAX_USER_SEQ_LEN)
    blocks_per_user_sliding = sliding_cache_len // BLOCK_SIZE  # 16
    full_pool_blocks = B * blocks_per_user_max  # 4096
    scratch_full_id = full_pool_blocks
    total_blocks_full = full_pool_blocks + 1
    total_blocks_sliding = (B + 1) * blocks_per_user_sliding

    sk_cap = min(PV_SK_CAP_REQUEST, MAX_USER_SEQ_LEN)
    sk_cap = (sk_cap // BLOCK_SIZE) * BLOCK_SIZE
    nblocks_full_pv = sk_cap // BLOCK_SIZE

    logger.info(
        f"config: tp={tp} B={B} P={P} layer_idx={layer_idx} layer_type={layer_type} "
        f"max_user_seq={MAX_USER_SEQ_LEN} block_size={BLOCK_SIZE} sk_cap={sk_cap} "
        f"sliding_w={sliding_cache_len}"
    )

    # 1. HF config + state dict.
    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config
    state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)

    # Sanity: ensure the chosen layer_idx matches the requested layer_type.
    actual_lt = model_args.layer_types[layer_idx]
    expected_lt = "sliding_attention" if is_sliding else "full_attention"
    assert actual_lt == expected_lt, (
        f"layer_idx={layer_idx} has type {actual_lt!r}, expected {expected_lt!r} for "
        f"layer_type={layer_type!r}. The default pattern is [S,S,S,S,S,G]; if your "
        f"checkpoint's layer_types differs, override _LAYER_IDX_BY_TYPE in this test."
    )

    # 2. Mesh + CCL.
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=2) if is_mesh and tp > 1 else None

    # 3. Build the single decoder layer (uses real weights via state_dict lookup).
    tensor_cache_path = os.environ.get("TT_CACHE_PATH") or None
    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_seq_len=MAX_USER_SEQ_LEN,
        max_local_batch_size=B,
    )
    if model_args.enable_moe_block:
        pytest.skip("MoE variants not supported by this single-layer harness")
    if getattr(model_args, "hidden_size_per_layer_input", 0):
        pytest.skip("PLI variants (E2B/E4B) not supported by this single-layer harness")

    _disable_fused_reduce_scatter(tt_layer)

    # 4. KV cache for this single layer, paged for the relevant attention type.
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)
    if is_sliding:
        page_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=total_blocks_sliding)
    else:
        page_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=total_blocks_full)
    kv_cache = init_kv_cache(
        mesh_device,
        attn_cfg,
        max_batch_size=B,
        max_seq_len=MAX_USER_SEQ_LEN,
        paged_attention_config=page_cfg,
        cache_dtype=ttnn.bfloat16,
    )
    tt_layer.self_attn.kv_cache = kv_cache

    # 5. RoPE 2D caches for this layer's type (decode embedding-lookup path).
    _, rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, MAX_USER_SEQ_LEN)
    cos_cache, sin_cache = rope_caches_2d[expected_lt]

    # 6. Slot state — every slot active, cur_pos in safe spec-eligibility window.
    cur_pos_base = 100
    rng = torch.Generator().manual_seed(0)
    slots: List[dict] = []
    free_full = list(range(full_pool_blocks))
    for i in range(B):
        n_pages = (cur_pos_base + P + BLOCK_SIZE - 1) // BLOCK_SIZE
        full_pages = torch.tensor([free_full.pop(0) for _ in range(n_pages)], dtype=torch.int32)
        sliding_pages = torch.arange(i * blocks_per_user_sliding, (i + 1) * blocks_per_user_sliding, dtype=torch.int32)
        slots.append({"cur_pos": cur_pos_base, "full_pages": full_pages, "sliding_pages": sliding_pages})

    # 7. Build packed-spec device buffers.
    H_local = model_args.num_attention_heads // tp

    # tokens / position_idx — position_idx is the only one the attention path uses;
    # tokens isn't read by the layer (no embedding in this test), so we omit it.
    position_idx_torch = torch.empty(1, B * P, dtype=torch.int32)
    for r in range(B):
        for p in range(P):
            position_idx_torch[0, r * P + p] = cur_pos_base + p
    position_idx_dev = _alloc(mesh_device, position_idx_torch, ttnn.uint32)

    # kv_write_idxs (full) + kv_write_idxs_sliding — one int32 [B] tensor per packed pos.
    W = sliding_cache_len
    kv_write_full = [torch.full((B,), cur_pos_base + p, dtype=torch.int32) for p in range(P)]
    kv_write_sliding = [torch.tensor([(cur_pos_base + p) % W] * B, dtype=torch.int32) for p in range(P)]
    kv_write_idxs_dev = [_alloc(mesh_device, t, ttnn.int32) for t in kv_write_full]
    kv_write_idxs_sliding_dev = [_alloc(mesh_device, t, ttnn.int32) for t in kv_write_sliding]

    # Per-slot page tables: real pages first, scratch padding after.
    if is_sliding:
        # Sliding layer reads page_table_sliding. Use the slot's sliding ring.
        pt_sliding = torch.zeros(B, blocks_per_user_sliding, dtype=torch.int32)
        for i, s in enumerate(slots):
            pt_sliding[i] = s["sliding_pages"]
        page_table_dev = None
        page_table_sliding_dev = _alloc(mesh_device, pt_sliding, ttnn.int32)
    else:
        # Full layer reads page_table. Build a [B, nblocks_full_pv] table padded with scratch.
        pt_full = torch.full((B, nblocks_full_pv), scratch_full_id, dtype=torch.int32)
        for i, s in enumerate(slots):
            n = int(s["full_pages"].shape[0])
            m = min(n, nblocks_full_pv)
            pt_full[i, :m] = s["full_pages"][:m]
        page_table_dev = _alloc(mesh_device, pt_full, ttnn.int32)
        page_table_sliding_dev = None

    # Causal attn_mask for the chosen type only — [B, 1, H_local*P, S_k] head-major.
    NEG = -1e9
    S_k = W if is_sliding else sk_cap
    mask_torch = torch.full((B, 1, H_local * P, S_k), NEG, dtype=torch.float32)
    for h in range(H_local):
        for p in range(P):
            mask_torch[:, 0, h * P + p, : cur_pos_base + p + 1] = 0.0
    attn_mask_dev = _alloc(mesh_device, mask_torch.to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # 8. Hidden states — random bf16 at the layer input, TILE layout [1, 1, B*P, hidden].
    hidden_torch = torch.randn(1, 1, B * P, model_args.hidden_size, dtype=torch.float32, generator=rng).to(
        torch.bfloat16
    )
    hidden_dev = _alloc(mesh_device, hidden_torch, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # 9. Build the packed spec dict the attention path expects.
    spec = {
        "p": P,
        "position_idx": position_idx_dev,
        "kv_write_idxs": kv_write_idxs_dev,
        "kv_write_idxs_sliding": kv_write_idxs_sliding_dev,
        "attn_mask": {expected_lt: attn_mask_dev},
    }

    # 10. Run the layer directly. No trace.
    ttnn.synchronize_device(mesh_device)
    out_dev = tt_layer(
        hidden_dev,
        rope_mats=(cos_cache, sin_cache),
        position_idx=None,
        page_table=page_table_dev,
        kv_cache=kv_cache,
        is_decode=True,
        page_table_sliding=page_table_sliding_dev,
        packed=spec,
    )
    ttnn.synchronize_device(mesh_device)

    # 11. Shape + finite-value assertions.
    out_torch = ttnn.to_torch(ttnn.get_device_tensors(out_dev)[0] if is_mesh else out_dev).float()
    # The decoder layer returns [1, 1, B*P, hidden_size] (or [B*P, hidden_size] depending on op chain).
    flat = out_torch.reshape(-1, model_args.hidden_size)
    assert flat.shape == (B * P, model_args.hidden_size), (
        f"layer output shape {tuple(out_torch.shape)} → flat {tuple(flat.shape)} != "
        f"expected ({B*P}, {model_args.hidden_size})"
    )
    assert torch.isfinite(flat).all(), "layer output has non-finite values"
    out_norm = flat.norm().item()
    assert out_norm > 0.0, "layer output norm is zero — model may not have executed"

    logger.info(f"OK {layer_type}: out shape={tuple(out_torch.shape)} norm={out_norm:.2f}")


# Drafter path (default mirrors server.py's _SPEC_DRAFTER_PATH).
_DRAFTER_PATH = os.environ.get("GEMMA4_DRAFTER_PATH", "/mnt/nas/gemma-31b-assistant")
_DRAFTER_CACHE = os.environ.get("GEMMA4_DRAFTER_CACHE_DIR") or (
    os.path.join(os.environ["TT_CACHE_PATH"], "tensor_cache_assistant_31b_bf16")
    if os.environ.get("TT_CACHE_PATH")
    else None
)


# Pin to the production 1x4 TP submesh (single pass; no (1,1)/(1,2)/(1,8) fan-out).
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.timeout(86400)
def test_single_drafter_step(mesh_device):
    """One drafter forward — no target model, no verify, no loop, no trace.

    Loads the MTP drafter (``Gemma4AssistantModel``) and calls
    ``drafter.forward`` exactly once with random doubled-hidden input + random
    shared K/V tensors + random RoPE. Asserts the (out_hidden, logits) output
    shapes match the drafter config.
    """
    if not os.path.isdir(_DRAFTER_PATH):
        pytest.skip(f"drafter weights not available at {_DRAFTER_PATH} (set GEMMA4_DRAFTER_PATH)")

    from models.demos.gemma4_cody.server.speculative import SpeculativeDecoder

    torch.manual_seed(0)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    B = DECODE_BATCH
    KV_LEN = 64  # multiple of SDPA k_chunk_size

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=2) if is_mesh and tp > 1 else None

    spec = SpeculativeDecoder(
        mesh_device=mesh_device,
        num_slots=B,
        num_drafts=1,
        assistant_path=_DRAFTER_PATH,
        drafter_cache_dir=_DRAFTER_CACHE,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
    )
    cfg = spec.drafter_config
    logger.info(
        f"drafter cfg: backbone_hidden={cfg.backbone_hidden_size} vocab={cfg.vocab_size} "
        f"layers={cfg.num_hidden_layers} layer_types={cfg.layer_types} "
        f"sliding(nkv={cfg.num_key_value_heads}, hd={cfg.head_dim}) "
        f"full(nkv={cfg.num_global_key_value_heads}, hd={cfg.global_head_dim})"
    )

    def _tile_bf16(t_torch):
        return _alloc(mesh_device, t_torch.to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Doubled hidden input — [embed(token) ‖ target_last_hidden]. Random here.
    target_hidden = _tile_bf16(torch.randn(1, 1, B, 2 * cfg.backbone_hidden_size))

    # Shared KV — one (K, V) pair per layer_type, matching the drafter forward's
    # non-paged signature. K, V are PER-DEVICE shape [B, nkv_local, kv_len, head_dim].
    # nkv_local = max(1, nkv // tp) — when nkv < tp each device holds one
    # GQA-assigned KV head; otherwise the KV heads are sharded across devices.
    # GQA requires Q_local (=num_attention_heads//tp) to be a multiple of nkv_local.
    nkv_sliding_local = max(1, cfg.num_key_value_heads // tp)
    nkv_full_local = max(1, cfg.num_global_key_value_heads // tp)
    K_swa = _tile_bf16(torch.randn(B, nkv_sliding_local, KV_LEN, cfg.head_dim))
    V_swa = _tile_bf16(torch.randn(B, nkv_sliding_local, KV_LEN, cfg.head_dim))
    K_full = _tile_bf16(torch.randn(B, nkv_full_local, KV_LEN, cfg.global_head_dim))
    V_full = _tile_bf16(torch.randn(B, nkv_full_local, KV_LEN, cfg.global_head_dim))
    shared_kv = {
        "sliding_attention": (K_swa, V_swa),
        "full_attention": (K_full, V_full),
    }

    # Per-slot RoPE (one row per slot — same per-row constant-position rule the
    # server uses: drafter locks to cur_pos for the whole round).
    cos_full = _tile_bf16(torch.randn(1, 1, B, cfg.global_head_dim))
    sin_full = _tile_bf16(torch.randn(1, 1, B, cfg.global_head_dim))
    cos_swa = _tile_bf16(torch.randn(1, 1, B, cfg.head_dim))
    sin_swa = _tile_bf16(torch.randn(1, 1, B, cfg.head_dim))

    cur_pos = _alloc(mesh_device, torch.full((B,), KV_LEN - 1, dtype=torch.int32), ttnn.int32)

    # ONE drafter forward.
    ttnn.synchronize_device(mesh_device)
    out_hidden, logits = spec.drafter.forward(
        target_last_hidden=target_hidden,
        shared_kv=shared_kv,
        cos_pos_full=cos_full,
        sin_pos_full=sin_full,
        cos_pos_sliding=cos_swa,
        sin_pos_sliding=sin_swa,
        cur_pos_tensor=cur_pos,
    )
    ttnn.synchronize_device(mesh_device)

    # Shape assertions.
    hidden_torch = ttnn.to_torch(ttnn.get_device_tensors(out_hidden)[0] if is_mesh else out_hidden).float()
    logits_torch = ttnn.to_torch(ttnn.get_device_tensors(logits)[0] if is_mesh else logits).float()

    assert hidden_torch.reshape(-1, cfg.backbone_hidden_size).shape == (
        B,
        cfg.backbone_hidden_size,
    ), f"drafter out_hidden shape {tuple(hidden_torch.shape)} != [..., {B}, {cfg.backbone_hidden_size}]"
    # logits are TP-sharded on the vocab dim — each device holds vocab/tp.
    expected_vocab_per_dev = cfg.vocab_size // tp
    flat_logits = logits_torch.reshape(-1, logits_torch.shape[-1])
    assert flat_logits.shape == (B, expected_vocab_per_dev), (
        f"drafter logits shape {tuple(logits_torch.shape)} → flat {tuple(flat_logits.shape)} != "
        f"[{B}, {expected_vocab_per_dev}] (vocab={cfg.vocab_size}, tp={tp})"
    )
    assert torch.isfinite(hidden_torch).all(), "drafter out_hidden has non-finite values"
    assert torch.isfinite(flat_logits).all(), "drafter logits have non-finite values"
    assert hidden_torch.norm().item() > 0.0, "drafter out_hidden norm is zero"
    assert flat_logits.norm().item() > 0.0, "drafter logits norm is zero"

    logger.info(
        f"OK drafter: hidden={tuple(hidden_torch.shape)} (norm={hidden_torch.norm():.2f}), "
        f"logits={tuple(logits_torch.shape)} (norm={flat_logits.norm():.2f}), top1[slot0]="
        f"{int(flat_logits[0].argmax().item())}"
    )
