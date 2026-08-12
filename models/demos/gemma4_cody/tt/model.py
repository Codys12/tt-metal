# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Model — full on-device implementation following gpt-oss pattern.

Architecture:
- 30 decoder layers with [5 sliding, 1 full] x 5 pattern
- Two RoPE configs: sliding (head_dim=256, theta=10k) and global (head_dim=512, theta=1M)
- Embedding scaled by sqrt(hidden_size)
- final_logit_softcapping = 30.0
- tie_word_embeddings = True

Supports both prefill and decode modes with paged attention.
Compatible with tt_transformers Generator interface.
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.demos.gemma4_cody.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4_cody.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4_cody.tt.rms_norm import RMSNorm
from models.demos.gemma4_cody.utils.general_utils import cached_tensor_placeholder, get_cache_file_name
from models.demos.gemma4_cody.utils.substate import substate


def _compute_per_device_vocab(vocab_size, num_tp):
    """Per-device vocab width: tile-aligned then rounded to next power of 2.

    Power-of-2 rounding enables ttnn.topk's multi-core bitonic sort.
    Must match between LM head weight padding and sampling args.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def create_rope_caches(mesh_device, hf_config, max_seq_len):
    """Create HF-format cos/sin caches for both sliding and global layer types.

    Returns:
        caches_4d: dict mapping layer_type -> (cos_tt, sin_tt) [1,1,max_seq_len,head_dim] for prefill
        caches_2d: dict mapping layer_type -> (cos_tt, sin_tt) [max_seq_len,head_dim] for decode embedding lookup
    """
    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    caches_4d = {}
    caches_2d = {}
    for layer_type in dict.fromkeys(hf_config.layer_types):
        cos, sin = _create_rope_cache_tensors(hf_config, max_seq_len, layer_type)
        # cos, sin: [1, max_seq_len, head_dim]

        # 4D for prefill: [1, 1, max_seq_len, head_dim]
        cos_4d = ttnn.from_torch(
            cos.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_4d = ttnn.from_torch(
            sin.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches_4d[layer_type] = (cos_4d, sin_4d)

        # 2D for decode embedding lookup: [max_seq_len, head_dim]
        cos_2d = ttnn.from_torch(
            cos.squeeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_2d = ttnn.from_torch(
            sin.squeeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches_2d[layer_type] = (cos_2d, sin_2d)

    return caches_4d, caches_2d


def _create_rope_cache_tensors(hf_config, max_seq_len, layer_type):
    """Create Gemma4 RoPE caches without invoking HF's Torch rotary forward path."""
    rope_params = hf_config.rope_parameters[layer_type]
    rope_type = rope_params["rope_type"]

    if rope_type == "default":
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        inv_freq = _default_rope_inv_freq(head_dim, rope_params["rope_theta"])
    elif rope_type == "proportional":
        head_dim = getattr(hf_config, "global_head_dim", None) or getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )
        inv_freq = _proportional_rope_inv_freq(head_dim, rope_params)
    else:
        raise ValueError(f"Unsupported Gemma4 RoPE type for offline cache generation: {rope_type!r}")

    positions = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1).astype(np.float32, copy=False)
    cos_np = np.ascontiguousarray(np.cos(emb).astype(np.float32, copy=False).reshape(1, max_seq_len, emb.shape[-1]))
    sin_np = np.ascontiguousarray(np.sin(emb).astype(np.float32, copy=False).reshape(1, max_seq_len, emb.shape[-1]))
    cos = torch.from_numpy(cos_np)
    sin = torch.from_numpy(sin_np)
    return cos, sin


def _default_rope_inv_freq(head_dim, theta):
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    return (1.0 / (np.float32(theta) ** exponent)).astype(np.float32, copy=False)


def _proportional_rope_inv_freq(head_dim, rope_params):
    theta = np.float32(rope_params["rope_theta"])
    factor = np.float32(rope_params.get("factor", 1.0))
    rope_proportion = rope_params.get("partial_rotary_factor", 1.0)
    rope_angles = int(rope_proportion * head_dim // 2)

    rotated_exponent = np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / np.float32(head_dim)
    inv_freq = 1.0 / (theta**rotated_exponent)

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = np.concatenate((inv_freq, np.zeros(nope_angles, dtype=np.float32)), axis=0)

    return (inv_freq / factor).astype(np.float32, copy=False)


class Gemma4Model:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_seq_len=131072,
        max_local_batch_size=1,
        num_layers=None,
        paged_attention_config=None,
        paged_attention_config_sliding=None,
        create_kv_cache=True,
        kv_cache_dtype=ttnn.bfloat16,
        create_rope_cache=True,
        # Legacy parameters — ignored
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        self.max_seq_len = max_seq_len
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0
        n_layers = num_layers or hf_config.num_hidden_layers

        # KV sharing map: layers after (full_n_layers - num_kv_shared_layers) share KV
        # from the last non-shared layer of the same type
        full_n_layers = hf_config.num_hidden_layers
        num_kv_shared = getattr(hf_config, "num_kv_shared_layers", 0) or 0
        first_shared_idx = full_n_layers - num_kv_shared
        self.kv_shared_layer_map = {}  # layer_idx -> source_layer_idx
        if num_kv_shared > 0 and first_shared_idx < n_layers:
            prev_layers = hf_config.layer_types[:first_shared_idx]
            for i in range(first_shared_idx, n_layers):
                lt = hf_config.layer_types[i]
                if lt in prev_layers:
                    source = len(prev_layers) - 1 - list(prev_layers)[::-1].index(lt)
                    if source < n_layers:  # Source must be within our layer range
                        self.kv_shared_layer_map[i] = source
            if self.kv_shared_layer_map:
                logger.info(f"KV sharing enabled: {len(self.kv_shared_layer_map)} layers share KV from earlier layers")

        # RoPE caches per layer type (sliding vs global)
        # Needs real HF text config (set by create_tt_model via _hf_text_config)
        hf_text_config = getattr(hf_config, "_hf_text_config", None)
        if hf_text_config is not None and create_rope_cache:
            logger.info("Creating RoPE caches")
            self.rope_caches, self.rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
            logger.info("RoPE caches created")
        else:
            # Fallback: no automatic RoPE — caller must pass rope_mats explicitly
            self.rope_caches = {}
            self.rope_caches_2d = {}

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        if state_dict and "model.language_model.embed_tokens.weight" in state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
        elif state_dict and "model.embed_tokens.weight" in state_dict:
            embed_key = "model.embed_tokens.weight"
        else:
            embed_key = None

        if embed_key and state_dict:
            logger.info("Loading embedding and LM head weights")
            embed_cache_name = get_cache_file_name(tensor_cache_path, f"embed_tokens.weight{tp_suffix}")
            lm_head_cache_name = get_cache_file_name(tensor_cache_path, f"lm_head.weight{tp_suffix}")
            embed_weight = None
            embed_tensor = cached_tensor_placeholder(embed_cache_name, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            lm_head_tensor = cached_tensor_placeholder(lm_head_cache_name, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            if embed_tensor is None or lm_head_tensor is None:
                embed_weight = state_dict[embed_key]

            # Embedding: column-parallel (shard hidden dim across TP devices)
            # Each device holds [vocab, hidden/TP]; all-gather after lookup.
            if tp > 1:
                embed_mapper = mesh_config.column_parallel(mesh_device)
            else:
                embed_mapper = replicate
            if embed_tensor is None:
                embed_tensor = embed_weight.unsqueeze(0).unsqueeze(0)
            self.embedding_weight = ttnn.as_tensor(
                embed_tensor,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=embed_mapper,
                cache_file_name=embed_cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # LM head (tied with embeddings): column-parallel (shard vocab dim)
            # Each device holds [hidden, vocab/TP]; all-gather logits after softcapping.
            if lm_head_tensor is None:
                lm_head_tensor = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            if tp > 1:
                lm_mapper = mesh_config.column_parallel(mesh_device)
            else:
                lm_mapper = replicate
            # Always bfloat16 for LM head — bfloat8_b is too lossy for 262k-vocab argmax
            self.lm_head_weight = ttnn.as_tensor(
                lm_head_tensor,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=lm_mapper,
                cache_file_name=lm_head_cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logger.info("Embedding and LM head weights loaded")
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Per-layer input embeddings (E2B/E4B) — kept as CPU torch tensors for computation
        # Also store embedding weight reference for decode per-layer input
        self._embed_weight_cpu = None
        self._embed_state_dict = state_dict
        self._embed_key = embed_key
        if embed_key and state_dict:
            self._embed_weight_cpu = embed_weight
        self.per_layer_input_weights = {}
        self._per_layer_input_weight_keys = None
        if self.hidden_size_per_layer_input and state_dict:
            pli_size = self.hidden_size_per_layer_input
            # Try both key formats
            for prefix in ["model.language_model.", "model."]:
                pli_embed_key = f"{prefix}embed_tokens_per_layer.weight"
                pli_proj_key = f"{prefix}per_layer_model_projection.weight"
                pli_norm_key = f"{prefix}per_layer_projection_norm.weight"
                if pli_embed_key in state_dict:
                    self._per_layer_input_weight_keys = {
                        "embed_tokens_per_layer": pli_embed_key,
                        "per_layer_model_projection": pli_proj_key,
                        "per_layer_projection_norm": pli_norm_key,
                    }
                    self.per_layer_input_scale = 2.0**-0.5
                    self.per_layer_model_projection_scale = hf_config.hidden_size**-0.5
                    self.per_layer_embed_scale = pli_size**0.5

                    logger.info(f"Per-layer input embeddings available (pli_size={pli_size})")
                    break

        # Decoder layers (each creates its own KV cache if requested)
        self.layers = []
        for i in range(n_layers):
            logger.info(f"Loading Gemma4 decoder layer {i + 1}/{n_layers}")
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                max_seq_len=max_seq_len,
                max_local_batch_size=max_local_batch_size,
            )
            # Create KV cache for non-shared layers only
            # Shared layers will use their source layer's KV cache
            if create_kv_cache and i not in self.kv_shared_layer_map:
                from models.demos.gemma4_cody.tt.attention.kv_cache import init_kv_cache

                attn_cfg = Gemma4AttentionConfig(hf_config, i)
                # Pick the per-layer-type paged config so sliding layers allocate
                # only W tokens of cache per slot instead of full max_seq_len.
                layer_paged_cfg = paged_attention_config
                if attn_cfg.is_sliding and paged_attention_config_sliding is not None:
                    layer_paged_cfg = paged_attention_config_sliding
                kv_cache = init_kv_cache(
                    mesh_device=mesh_device,
                    config=attn_cfg,
                    max_batch_size=max_local_batch_size,
                    max_seq_len=max_seq_len,
                    paged_attention_config=layer_paged_cfg,
                    cache_dtype=kv_cache_dtype,
                )
                layer.self_attn.kv_cache = kv_cache
            self.layers.append(layer)
            logger.info(f"Loaded Gemma4 decoder layer {i + 1}/{n_layers}")

        # Extract KV caches for external access (Generator interface)
        # Shared layers point to their source layer's cache
        self.tt_kv_cache = []
        for i, layer in enumerate(self.layers):
            if i in self.kv_shared_layer_map:
                source_idx = self.kv_shared_layer_map[i]
                self.tt_kv_cache.append(self.layers[source_idx].self_attn.kv_cache)
            else:
                self.tt_kv_cache.append(layer.self_attn.kv_cache)

        # Per-layer staging buffers for the loop-free packed KV write. Mirrors
        # tt_kv_cache: non-shared layers own a buffer, shared layers reuse their
        # source layer's. The attention forward reads ``self.kv_staging`` (set
        # here for every layer), so no model.forward threading is needed.
        self.tt_kv_staging = [None] * len(self.layers)
        if create_kv_cache:
            from models.demos.gemma4_cody.tt.attention.kv_cache import PV_HOT_BLOCKS, init_kv_staging

            stage_block_size = paged_attention_config.block_size if paged_attention_config else 64
            for i, layer in enumerate(self.layers):  # pass 1: non-shared own a buffer
                if i not in self.kv_shared_layer_map:
                    attn_cfg = Gemma4AttentionConfig(hf_config, i)
                    layer.self_attn.kv_staging = init_kv_staging(
                        mesh_device=mesh_device,
                        config=attn_cfg,
                        max_batch_size=max_local_batch_size,
                        block_size=stage_block_size,
                        blk=PV_HOT_BLOCKS,
                        cache_dtype=kv_cache_dtype,
                    )
            for i, layer in enumerate(self.layers):  # pass 2: shared alias source + collect
                if i in self.kv_shared_layer_map:
                    layer.self_attn.kv_staging = self.layers[self.kv_shared_layer_map[i]].self_attn.kv_staging
                self.tt_kv_staging[i] = layer.self_attn.kv_staging

        # Last-of-each-layer-type index. Consumed by the speculative-decode
        # drafter, which reads the target's KV at the deepest layer of each
        # layer_type (sliding_attention / full_attention). Computed once.
        self.last_layer_of_type: dict = {}
        for i, lt in enumerate(hf_config.layer_types):
            self.last_layer_of_type[lt] = i

        # Final norm
        if state_dict and "model.language_model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.language_model.norm")
        elif state_dict and "model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.norm")
        else:
            norm_state = {}

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=norm_state,
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
            enable_sharded_decode=True,
        )
        logger.info("Final norm loaded")

        # On-device sampling (greedy/top-k/top-p) — avoids reading full vocab logits to CPU
        self.sampling = None
        if is_mesh and tp > 1:
            per_device_padded = _compute_per_device_vocab(hf_config.vocab_size, tp)
            if per_device_padded <= 64 * 1024:
                self.sampling = SamplingGenerator(
                    args=self._make_sampling_args(hf_config, mesh_device, tp),
                    mesh_device=mesh_device,
                    tt_ccl=None,
                    enable_internal_trace=False,
                )
                logger.info(
                    f"On-device sampling initialized (vocab={hf_config.vocab_size}, per_device={per_device_padded})"
                )

    @staticmethod
    def _make_sampling_args(hf_config, mesh_device, tp):
        """Create minimal args object for SamplingGenerator/TTSampling."""

        class _Args:
            pass

        args = _Args()
        args.vocab_size = hf_config.vocab_size
        per_device_vocab = _compute_per_device_vocab(args.vocab_size, tp)
        args.padded_vocab_size = per_device_vocab * tp
        args.cluster_shape = tuple(mesh_device.shape)
        args.sampling_all_gather_axis = 1  # gather across TP (column) axis
        args.sampling_dp = 1
        args.num_devices = mesh_device.get_num_devices()
        args.is_galaxy = mesh_device.shape[0] > 1
        args.model_config = {}
        args.use_topk_logprobs = False
        return args

    def _compute_per_layer_inputs(self, input_ids_torch, embeds_torch):
        """Compute per-layer input embeddings on CPU (E2B/E4B).

        Returns list of [1, seq_len, pli_size] tensors, one per layer, or None.
        Returns None if input_ids_torch or embeds_torch are not provided (e.g. trace mode).
        """
        if not self.hidden_size_per_layer_input or not self._per_layer_input_weight_keys:
            return None
        if input_ids_torch is None or embeds_torch is None:
            return None

        import torch.nn.functional as F

        if not self.per_layer_input_weights:
            self.per_layer_input_weights = {
                name: self._embed_state_dict[key] for name, key in self._per_layer_input_weight_keys.items()
            }
        w = self.per_layer_input_weights
        pli_size = self.hidden_size_per_layer_input
        n_layers = len(self.layers)
        # The per-layer embedding weight has ALL layers baked in
        # Infer full layer count from the weight shape, not the (possibly overridden) config
        embed_w = w["embed_tokens_per_layer"]  # [vocab_pli, full_n_layers * pli_size]
        full_n_layers = embed_w.shape[-1] // pli_size

        # 1. Per-layer token embedding: embed_tokens_per_layer(input_ids)
        pli_embed = F.embedding(input_ids_torch.long(), embed_w) * self.per_layer_embed_scale
        pli_embed = pli_embed.reshape(*input_ids_torch.shape, full_n_layers, pli_size)

        # 2. Projection from main embeddings
        proj_w = w["per_layer_model_projection"]  # [full_n_layers * pli_size, hidden]
        pli_proj = F.linear(embeds_torch.float(), proj_w.float()) * self.per_layer_model_projection_scale
        pli_proj = pli_proj.reshape(*embeds_torch.shape[:-1], full_n_layers, pli_size)

        # 3. Norm the projection
        norm_w = w["per_layer_projection_norm"]  # [pli_size]
        eps = self.hf_config.rms_norm_eps
        pli_proj_f = pli_proj.float()
        var = pli_proj_f.pow(2).mean(-1, keepdim=True)
        pli_proj = (pli_proj_f * torch.rsqrt(var + eps) * norm_w.float()).to(pli_proj.dtype)

        # 4. Combine: (projection + embed) * scale
        per_layer_inputs = (pli_proj + pli_embed.float()) * self.per_layer_input_scale

        # Return as list of per-layer tensors
        return [per_layer_inputs[:, :, i, :].to(torch.bfloat16) for i in range(n_layers)]

    def _get_rope_mats(self, layer_idx, seq_len=None, for_decode=False):
        """Get (cos, sin) for a given layer.

        Args:
            seq_len: If set, slice 4D cache to this length (prefill).
            for_decode: If True, return 2D caches [max_seq_len, head_dim] for embedding lookup.
        """
        layer_type = self.hf_config.layer_types[layer_idx]
        if for_decode:
            return self.rope_caches_2d[layer_type]
        cos, sin = self.rope_caches[layer_type]
        if seq_len is not None:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        return (cos, sin)

    def configure_aux_taps(self, layer_indices) -> None:
        """Register which layer outputs to expose when ``return_aux_hidden=True``.

        ``layer_indices`` are absolute indices into the full layer stack (the
        DFlash drafter's ``aux_hidden_layers``, already ``-1``-shifted). They are
        clamped to the loaded layer count at forward time (``--num-layers`` may
        load fewer), and the returned tap list preserves this order — so the
        DFlash ``fc`` always sees its K taps concatenated in the trained order
        even if clamping collapses two onto the same loaded layer.
        """
        self._aux_tap_layers = tuple(int(i) for i in layer_indices)

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_caches=None,
        is_decode=True,
        token_index=None,
        input_ids_torch=None,
        embeds_torch=None,
        pli_device_tensors=None,
        position_idx_cache=None,
        pli_combined=None,
        page_table_sliding=None,
        position_idx_cache_sliding_write=None,
        position_idx_cache_sliding_sdpa=None,
        skip_logits=False,
        return_hidden_state=False,
        return_aux_hidden=False,
        packed=None,
    ):
        """
        Forward pass through decoder layers + final norm + lm_head + softcapping.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device (post-embedding)
            rope_mats: (cos, sin) override, or dict {layer_type: (cos, sin)} for pre-sliced decode
            position_idx: decode position tensor ([1,32] uint32 for embedding RoPE, or [1] int32 legacy)
            page_table: paged attention table
            kv_caches: list of [k, v] per layer, or None (uses self.tt_kv_cache)
            is_decode: True for decode, False for prefill
            token_index: int for decode RoPE slicing (None when using embedding-based RoPE)
            input_ids_torch: CPU tensor of input_ids for per-layer input computation (E2B)
            embeds_torch: CPU tensor of embeddings for per-layer input projection (E2B)
            pli_device_tensors: optional list of pre-computed PLI device tensors (trace mode)
            position_idx_cache: optional [batch] int32 tensor for KV cache update (when position_idx is uint32)
            pli_combined: optional [1,1,n_layers,pli_size] device tensor of pre-computed PLI (decode)
            skip_logits: if True, skip lm_head + softcap + all-gather and return the
                post-norm hidden state instead of logits. The caller (typically the
                continuous-batching server) discards prefill logits, so this avoids
                materialising the [1, 1, seq_len, vocab] all-gather output at long
                sequence lengths.
        """
        seq_len = hidden_states.shape[2]
        caches = kv_caches or self.tt_kv_cache

        # Compute per-layer inputs (E2B/E4B)
        # Decode: pre-computed on host via compute_host_embeddings (pli_combined)
        # Prefill: computed on CPU from input_ids_torch / embeds_torch
        pli_combined_tt = None
        per_layer_inputs = None
        if pli_combined is not None:
            pli_combined_tt = pli_combined
        elif pli_device_tensors is not None:
            pass  # Pre-computed device tensors provided externally
        else:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids_torch, embeds_torch)

        is_mesh = hasattr(self.mesh_device, "shape")

        # Determine which layers are KV sources (their K/V will be shared)
        kv_source_indices = set(self.kv_shared_layer_map.values()) if not is_decode else set()
        # Store K/V from source layers for sharing during prefill
        shared_kv_store = {}  # source_layer_idx -> (tt_k, tt_v) kept alive on device

        # Aux-hidden taps for the DFlash drafter: clone the residual-stream
        # hidden after the configured layers (post-`layer_scalar` output, == the
        # next layer's input residual, which is what DFlash conditions on). Clone
        # (not alias) — the next layer reassigns/may free `hidden_states`. The
        # clone outputs are allocated as part of the (decode / packed-verify)
        # trace's static plan, so this is trace-safe, same as the final-hidden
        # return below. ``configure_aux_taps`` sets ``self._aux_tap_layers``.
        aux_layers = getattr(self, "_aux_tap_layers", ()) if return_aux_hidden else ()
        n_loaded = len(self.layers)
        aux_clamped = [min(int(l), n_loaded - 1) for l in aux_layers]
        aux_capture_set = set(aux_clamped)
        aux_clones: dict = {}

        for i, layer in enumerate(self.layers):
            # Per-layer RoPE: sliding and global layers have different cos/sin
            if rope_mats is not None:
                if isinstance(rope_mats, dict):
                    # Dict mapping layer_type -> (cos, sin) — pre-sliced for trace decode
                    layer_type = self.hf_config.layer_types[i]
                    layer_rope = rope_mats[layer_type]
                else:
                    layer_rope = rope_mats  # Single (cos, sin) override (backward compat / tests)
            elif is_decode:
                # Decode: return 2D caches for on-device embedding lookup
                layer_rope = self._get_rope_mats(i, for_decode=True)
            else:
                layer_rope = self._get_rope_mats(i, seq_len=seq_len)

            # Convert per-layer input to device tensor if available
            pli_tt = None
            if pli_combined_tt is not None:
                # On-device decode: slice layer i from combined [1, 1, n_layers, pli_size]
                pli_tt = pli_combined_tt[:, :, i : i + 1, :]
            elif pli_device_tensors is not None and i < len(pli_device_tensors):
                # Pre-computed device tensors (legacy trace mode)
                pli_tt = pli_device_tensors[i]
            elif per_layer_inputs is not None and i < len(per_layer_inputs):
                pli_4d = per_layer_inputs[i].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, pli_size]
                pli_tt = ttnn.from_torch(
                    pli_4d,
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
                )

            kv_cache = caches[i] if caches else None

            # KV sharing: determine if this layer shares or provides K/V
            shared_kv = None
            keep_kv = False
            is_kv_shared = i in self.kv_shared_layer_map
            if not is_decode and is_kv_shared:
                source_idx = self.kv_shared_layer_map[i]
                shared_kv = shared_kv_store.get(source_idx)
            elif not is_decode and i in kv_source_indices:
                keep_kv = True

            hidden_states = layer(
                hidden_states,
                rope_mats=layer_rope,
                position_idx=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                is_decode=is_decode,
                token_index=token_index,
                per_layer_input=pli_tt,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
                is_kv_shared=is_kv_shared,
                position_idx_cache=position_idx_cache,
                page_table_sliding=page_table_sliding,
                position_idx_cache_sliding_write=position_idx_cache_sliding_write,
                position_idx_cache_sliding_sdpa=position_idx_cache_sliding_sdpa,
                packed=packed,
            )

            if aux_capture_set and i in aux_capture_set:
                aux_clones[i] = ttnn.clone(hidden_states)

            # For KV source layers during prefill, capture the K/V from the attention
            # The K/V are kept alive on device (not deallocated) when keep_kv=True
            if keep_kv and layer.self_attn._last_kv is not None:
                shared_kv_store[i] = layer.self_attn._last_kv

        # Deallocate any stored shared K/V tensors
        for kv_pair in shared_kv_store.values():
            if kv_pair is not None:
                kv_pair[0].deallocate(True)
                kv_pair[1].deallocate(True)

        # Final norm
        hidden_states = self.norm.forward(hidden_states)

        # Skip the LM head entirely when the caller doesn't need logits (e.g. the
        # server's prefill, which discards them and reproduces the first token via
        # the decode trace). This avoids both the [1, 1, seq, vocab/tp] linear
        # output and the [1, 1, seq, vocab] all-gather, which are large at long
        # prompt lengths (e.g. 9 GiB/device at seq_len=16k for vocab=262144, bf16).
        if skip_logits:
            if return_aux_hidden:
                return hidden_states, [aux_clones[c] for c in aux_clamped]
            return hidden_states

        # Optionally keep the post-norm hidden state for downstream consumers
        # (the spec-decode drafter — empirically the drafter's pre_projection
        # wants the post-final-norm hidden; pre-norm measured worse).
        hidden_state_for_spec = hidden_states if return_hidden_state else None

        # LM head (column-parallel on vocab dim when TP > 1)
        if self.lm_head_weight is not None:
            logits = ttnn.linear(hidden_states, self.lm_head_weight)
            if hidden_state_for_spec is None:
                hidden_states.deallocate(True)
        else:
            logits = hidden_states
            # If logits IS hidden_states (no lm_head), share the reference.
            hidden_state_for_spec = hidden_states if return_hidden_state else None

        # Softcapping: tanh(logits / cap) * cap — element-wise, works on sharded vocab
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)

        # All-gather sharded vocab dim back to full vocab.
        # Skip when on-device sampling is active (decode) — sampling handles distributed top-k.
        if self.mesh_config is not None and self.mesh_config.tp > 1 and self.lm_head_weight is not None:
            if self.sampling is not None and is_decode:
                pass  # Sampling module handles TP-sharded logits directly
            else:
                from models.demos.gemma4_cody.tt.ccl import ccl_allgather

                logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        if return_aux_hidden:
            return logits, hidden_state_for_spec, [aux_clones[c] for c in aux_clamped]
        if return_hidden_state:
            return logits, hidden_state_for_spec
        return logits

    def embed_tokens(self, tokens):
        """Embed input tokens and scale by sqrt(hidden_size).

        Embedding is column-parallel (hidden dim sharded across TP devices).
        All-gather reconstructs full hidden dim after lookup.
        """
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        embeds = ttnn.mul(embeds, self.embed_scale)

        # All-gather sharded hidden dim back to full hidden
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            embeds = ttnn.unsqueeze_to_4D(embeds)
            from models.demos.gemma4_cody.tt.ccl import ccl_allgather

            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return embeds

    def compute_host_embeddings(self, token_id):
        """Compute token embedding + PLI entirely on CPU for a single decode token.

        Embedding and PLI are computed on host to keep them off the device trace
        on 1x1 mesh.

        Returns:
            (embeds, pli_combined) where:
            - embeds: torch.Tensor [1, 1, 1, hidden_size] bfloat16
            - pli_combined: torch.Tensor [1, 1, n_layers, pli_size] bfloat16, or None
        """
        import torch.nn.functional as F

        token_tensor = torch.tensor([[token_id]], dtype=torch.long)

        # Token embedding (mirrors embed_tokens but on CPU)
        if self._embed_weight_cpu is None:
            if self._embed_state_dict is None or self._embed_key is None:
                raise RuntimeError("CPU embedding weights not available")
            self._embed_weight_cpu = self._embed_state_dict[self._embed_key]
        embeds = F.embedding(token_tensor, self._embed_weight_cpu).float() * self.embed_scale

        # Per-layer input (E2B/E4B)
        pli_combined = None
        if self.hidden_size_per_layer_input and self._per_layer_input_weight_keys:
            pli_list = self._compute_per_layer_inputs(token_tensor.int(), embeds)
            if pli_list is not None:
                pli_combined = torch.stack(pli_list, dim=2)  # [1, 1, n_layers, pli_size]

        embeds = embeds.reshape(1, 1, 1, self.hidden_size).to(torch.bfloat16)
        return embeds, pli_combined

    # ── Generator-compatible interface ────────────────────────────────────

    def ttnn_prefill_forward(
        self,
        x,
        user_id=0,
        page_table=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        input_ids_torch=None,
        embeds_torch=None,
        page_table_sliding=None,
        skip_logits=False,
    ):
        """Prefill forward — matches tt_transformers Generator interface.

        With ``skip_logits=True`` the model returns post-norm hidden states
        instead of logits (no lm_head, no softcap, no all-gather). The
        ``get_last_token`` slice still applies and is now cheap because the
        last dim is ``hidden_size`` rather than ``vocab_size``.
        """
        seq_len = x.shape[-2]
        out = self(
            hidden_states=x,
            position_idx=None,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=False,
            input_ids_torch=input_ids_torch,
            embeds_torch=embeds_torch,
            page_table_sliding=page_table_sliding,
            skip_logits=skip_logits,
        )

        # Extract last token tile. Last dim is ``vocab_size`` when skip_logits=False
        # and ``hidden_size`` when skip_logits=True; ``out.shape[-1]`` covers both.
        if get_last_token != -1:
            out = ttnn.slice(
                out,
                (0, 0, get_last_token, 0),
                (1, 1, get_last_token + 32, out.shape[-1]),
            )

        return out

    def switch_mode(self, mode):
        """Generator compatibility — no prefetcher to reinitialize."""

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Create host tensors for one decode step, including pre-computed embedding + PLI.

        Called by Generator._capture_decode_trace_text and _decode_forward_trace_text.
        Returns tuple of host ttnn tensors that copy_host_to_device will transfer.

        Args:
            tokens: torch.Tensor [batch] of token IDs
            current_pos: torch.Tensor [batch] of current positions
            page_table: optional torch.Tensor [batch, max_blocks] page table
        """
        import torch.nn.functional as F

        is_mesh = hasattr(self.mesh_device, "shape")
        replicate = (
            ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh and self.mesh_device.get_num_devices() > 1 else None
        )

        # Host embedding + PLI (keeps these off the device trace)
        token_id = tokens[0].item()
        embeds, pli = self.compute_host_embeddings(token_id)

        embeds_tt = ttnn.from_torch(embeds, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)

        # Position: [1, 32] uint32 padded (for RoPE embedding lookup)
        pos = current_pos[0].item() if hasattr(current_pos, "item") else int(current_pos[0])
        pos_padded = F.pad(torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0)
        pos_tt = ttnn.from_torch(pos_padded, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate)

        # int32 position for KV cache update + SDPA
        pos_int32_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )

        # Page table
        page_table_tt = None
        if page_table is not None:
            page_table_tt = ttnn.from_torch(
                page_table[0:1] if page_table.dim() > 1 else page_table.unsqueeze(0),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            )

        # PLI
        pli_tt = None
        if pli is not None:
            pli_tt = ttnn.from_torch(
                pli.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
            )

        return (embeds_tt, pos_tt, pos_int32_tt, page_table_tt, pli_tt)

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Wrapper: prepare_decode_inputs_host + copy to device."""
        from models.tt_transformers.tt.common import copy_host_to_device

        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        return copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
        capture_sampling_trace=False,
        precomputed_pli=None,
        page_table_sliding=None,
        position_idx_cache_sliding_write=None,
        position_idx_cache_sliding_sdpa=None,
        return_hidden_state=False,
        return_aux_hidden=False,
    ):
        """Decode forward — matches tt_transformers Generator interface.

        x is a single-token embedding tensor, usually produced on device from token IDs.
        Generator calls: prepare_decode_inputs_host → copy_host_to_device → ttnn_decode_forward.

        Args:
            x: [1,1,1,hidden_size] ROW_MAJOR device tensor (pre-computed embedding).
            current_pos: [1,32] uint32 position tensor for RoPE embedding lookup.
            rot_mat_idxs: Unused (RoPE computed internally from current_pos).
            page_table: Optional paged attention table.
            kv_cache: Optional KV cache override.
            sampling_on_device: If True and self.sampling exists, sample on device.
            capture_sampling_trace: If True, return logits for split-trace sampling.
            precomputed_pli: Optional [1,1,n_layers,pli_size] device tensor for PLI models.
        """
        # Convert ROW_MAJOR host data to TILE on device
        input_embeds = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # RoPE: always use internal 2D caches with on-device embedding lookup
        token_index = None if self.rope_caches_2d else 0

        position_idx_cache = rot_mat_idxs  # Generator passes pos_int32 as rot_mat_idxs (3rd tuple element)

        aux_taps = None
        if return_hidden_state or return_aux_hidden:
            out = self(
                hidden_states=input_embeds,
                position_idx=current_pos,
                page_table=page_table,
                kv_caches=kv_cache,
                is_decode=True,
                token_index=token_index,
                position_idx_cache=position_idx_cache,
                pli_combined=ttnn.to_layout(precomputed_pli, ttnn.TILE_LAYOUT) if precomputed_pli is not None else None,
                page_table_sliding=page_table_sliding,
                position_idx_cache_sliding_write=position_idx_cache_sliding_write,
                position_idx_cache_sliding_sdpa=position_idx_cache_sliding_sdpa,
                return_hidden_state=True,
                return_aux_hidden=return_aux_hidden,
            )
            if return_aux_hidden:
                logits, final_hidden, aux_taps = out
            else:
                logits, final_hidden = out
        else:
            logits = self(
                hidden_states=input_embeds,
                position_idx=current_pos,
                page_table=page_table,
                kv_caches=kv_cache,
                is_decode=True,
                token_index=token_index,
                position_idx_cache=position_idx_cache,
                pli_combined=ttnn.to_layout(precomputed_pli, ttnn.TILE_LAYOUT) if precomputed_pli is not None else None,
                page_table_sliding=page_table_sliding,
                position_idx_cache_sliding_write=position_idx_cache_sliding_write,
                position_idx_cache_sliding_sdpa=position_idx_cache_sliding_sdpa,
            )
            final_hidden = None

        # On-device sampling
        if sampling_on_device and self.sampling is not None:
            if capture_sampling_trace:
                # Split-trace: return logits for separate sampling trace.
                if return_aux_hidden:
                    return logits, final_hidden, aux_taps
                return (logits, final_hidden) if return_hidden_state else logits
            batch_dim = logits.shape[2]
            if batch_dim < 32:
                logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
            tt_tokens, tt_log_probs = self.sampling.sample(logits, enable_trace=False)
            if return_aux_hidden:
                return tt_tokens, tt_log_probs, final_hidden, aux_taps
            if return_hidden_state:
                return tt_tokens, tt_log_probs, final_hidden
            return tt_tokens, tt_log_probs

        if return_aux_hidden:
            return logits, None, final_hidden, aux_taps
        if return_hidden_state:
            return logits, None, final_hidden
        return logits, None
