# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1, single-device Muse-Glimmer server with native DFlash decoding."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, GenerationConfig

import ttnn
from models.demos.muse_glimmer.server.protocol import (
    IncrementalMuseResponse,
    normalize_messages,
    parse_tokenizer_response,
    validate_reasoning_strength,
)
from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH, create_tt_model
from models.demos.muse_glimmer.tt.decode_core_config import MuseDecodeCoreConfig
from models.demos.muse_glimmer.tt.dflash import DFlashConfig, DFlashDrafter
from models.demos.muse_glimmer.tt.dflash.rope import build_rope_cache
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.prefetcher import Prefetcher

DEFAULT_ASSISTANT_PATH = (
    "/home/user/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/"
    "snapshots/2c86316d689027b91123638739743fef1d425233"
)
PHYSICAL_VERIFY_TOKENS = 32
KV_PAGE_SIZE = 64
PREFILL_ALIGNMENT = KV_PAGE_SIZE
PREFILL_CHUNK_SIZE = 2048
DFLASH_CACHE_CAPACITY = 2048
DEFAULT_TRACE_REGION_SIZE = 256_000_000
_TRACY_SIGNPOSTS = os.getenv("MUSE_TRACY_SIGNPOSTS") == "1"
if _TRACY_SIGNPOSTS:
    from tracy import signpost as _tracy_signpost


def _profile_signpost(header, message=None):
    if _TRACY_SIGNPOSTS:
        _tracy_signpost(header, message)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    accepted_drafts: list[int]
    elapsed_seconds: float
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    recipient: str | None = None
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    prefilled_prompt_tokens: int = 0
    chunked_prefill_tokens: int = 0
    packed_prefill_tokens: int = 0
    tokenization_seconds: float = 0.0
    prefill_seconds: float = 0.0
    cache_snapshot_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prefilled_prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        return self.ar_decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def ar_decode_tokens(self) -> int:
        # The first completion token is the bonus sampled by the final prefill
        # logits. Count only tokens produced after the AR replay loop starts.
        return max(0, len(self.token_ids) - 1)

    @property
    def ar_decode_tokens_per_second(self) -> float:
        return self.decode_tokens_per_second


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 32
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    reasoning_strength: str = "low"
    tool_namespace_descriptions: dict[str, str] | None = None
    current_date: str | None = None
    knowledge_cutoff: str | None = None
    stream: bool = False
    stream_options: dict | None = None


class Engine:
    """One request at a time; greedy decoding only."""

    def __init__(
        self,
        mesh_device,
        model_path: str = DEFAULT_MODEL_PATH,
        assistant_path: str = DEFAULT_ASSISTANT_PATH,
        max_seq_len: int = 512,
        assistant_cache_path: str | None = None,
    ):
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Muse Glimmer supports exactly one device")
        if max_seq_len % KV_PAGE_SIZE:
            raise ValueError(f"max_seq_len must be divisible by {KV_PAGE_SIZE}")
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.assistant_path = str(assistant_path)
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        generation_config = GenerationConfig.from_pretrained(self.model_path)
        eos = generation_config.eos_token_id
        self.eos_token_ids = {int(eos)} if isinstance(eos, int) else {int(token) for token in (eos or [])}

        self.model_args, self.model, _, _ = create_tt_model(
            mesh_device,
            max_seq_len=max_seq_len,
            model_path=self.model_path,
        )
        self.qkv_prefetcher = None
        self.decode_core_config = None
        self._ar_cores = None
        if os.getenv("MUSE_QKV_PREFETCH", "0") == "1":
            self._prefetch_mlp = os.getenv("MUSE_PREFETCH_MLP", "0") == "1"
            receivers_per_bank = int(os.getenv("MUSE_PREFETCH_RECEIVERS_PER_BANK", "3"))
            if receivers_per_bank not in (2, 3, 9):
                raise ValueError("Muse prefetch receivers per bank must be 2, 3, or 9")
            receiver_mapping = (
                MuseDecodeCoreConfig.full_bandwidth_sender_mapping()
                if receivers_per_bank == 9
                else MuseDecodeCoreConfig.isolated_sender_mapping(receivers_per_bank)
            )
            self._ar_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
            target_ar_cores = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(8, 9),
                    )
                ]
            )
            # The full target rectangle already contains every receiver core.
            target_consumer_cores = target_ar_cores
            self.qkv_prefetcher = Prefetcher(
                mesh_device,
                # Per layer: QKV, then MLP gate[3]/up[3].
                num_tensors=7 if self._prefetch_mlp else 1,
                num_layers=len(self.model.layers),
                num_receiver_cores=receivers_per_bank,
                receiver_mapping_override=receiver_mapping,
                model_name="Muse-Glimmer-30B",
                isolate_receiver_subdevice=False,
                worker_core_range_override=target_consumer_cores,
            )
            self.qkv_prefetcher.enable_performance_mode = os.getenv("MUSE_QKV_PREFETCH_PERFORMANCE_MODE", "1") == "1"
        self.dflash_config = DFlashConfig.from_hf_path(self.assistant_path)
        if self.dflash_config.block_size != 16:
            raise ValueError(f"Expected native DFlash block_size=16, got {self.dflash_config.block_size}")
        self.model.configure_aux_taps(self.dflash_config.target_layer_ids)
        decode_page_ids = torch.arange(max_seq_len // KV_PAGE_SIZE, dtype=torch.int32).reshape(1, -1)
        self.decode_page_table = self._to_tt(
            decode_page_ids.repeat(PHYSICAL_VERIFY_TOKENS, 1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        cache_dir = (
            Path(assistant_cache_path)
            if assistant_cache_path
            else (Path(self.assistant_path) / "tensor_cache_muse_glimmer_dflash_attn_bfp8_mlp_bfp4")
        )
        self.drafter = DFlashDrafter(
            mesh_device=mesh_device,
            config=self.dflash_config,
            cache_dir=str(cache_dir),
            attention_dtype=ttnn.bfloat8_b,
            mlp_dtype=ttnn.bfloat4_b,
            fc_dtype=ttnn.bfloat8_b,
            safetensors_dir=self.assistant_path,
            target_lm_head=self.model.lm_head_weight,
        )
        self.anchor_cache = self.drafter.init_anchor_cache()
        self.fixed_anchor_caches = self.drafter.alloc_fixed_anchor_caches(DFLASH_CACHE_CAPACITY)
        self._draft_noise = self._to_tt(
            torch.zeros(1, 1, self.dflash_config.block_size, self.dflash_config.hidden_size, dtype=torch.bfloat16)
        )
        self._draft_position_idx = self._to_tt(
            torch.zeros(1, self.dflash_config.block_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._draft_valid_len = self._to_tt(
            torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._draft_write_control = self._to_tt(
            torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._anchor_write_control = self._to_tt(
            torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        mask_columns = torch.arange(DFLASH_CACHE_CAPACITY).unsqueeze(0)
        mask_lengths = torch.arange(DFLASH_CACHE_CAPACITY + 1).unsqueeze(1)
        self._draft_mask_table = self._to_tt(
            torch.where(mask_columns < mask_lengths, 0.0, -1e9).to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT if self.qkv_prefetcher is not None else ttnn.TILE_LAYOUT,
        )
        # Embedding accepts tiled weights by inserting an untilize over the
        # device's full compute grid.  That implicit grid includes the QKV
        # prefetch sender cores and cannot execute while the decode sub-device
        # manager is resident.  Keep dedicated row-major lookup tables for AR:
        # the fused embedding then runs directly on its (small) worker grid.
        self._ar_rope_caches_2d = self.model.rope_caches_2d
        if self.qkv_prefetcher is not None:
            self._ar_rope_caches_2d = {
                layer_type: tuple(ttnn.to_layout(cache, ttnn.ROW_MAJOR_LAYOUT) for cache in caches)
                for layer_type, caches in self.model.rope_caches_2d.items()
            }
        self._draft_fixed_cos = self._to_tt(
            torch.zeros(1, 1, DFLASH_CACHE_CAPACITY, self.dflash_config.head_dim, dtype=torch.bfloat16)
        )
        self._draft_fixed_sin = self._to_tt(
            torch.zeros(1, 1, DFLASH_CACHE_CAPACITY, self.dflash_config.head_dim, dtype=torch.bfloat16)
        )
        self._draft_ids = self._to_tt(
            torch.zeros(1, 1, self.dflash_config.block_size - 1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
        )
        self._verify_hidden = self._to_tt(
            torch.zeros(1, 1, PHYSICAL_VERIFY_TOKENS, self.model.hidden_size, dtype=torch.bfloat16)
        )
        self._verify_position_idx = self._to_tt(
            torch.zeros(1, PHYSICAL_VERIFY_TOKENS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._append_position_idx = self._to_tt(
            torch.zeros(1, self.dflash_config.block_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._verify_rope = {
            layer_type: (
                self._to_tt(torch.zeros(1, PHYSICAL_VERIFY_TOKENS, 1, self.model_args.head_dim)),
                self._to_tt(torch.zeros(1, PHYSICAL_VERIFY_TOKENS, 1, self.model_args.head_dim)),
            )
            for layer_type in self.model.rope_caches_cpu
        }
        self._verify_cur_pos = self._to_tt(
            torch.zeros(PHYSICAL_VERIFY_TOKENS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._verify_ids = self._to_tt(
            torch.zeros(1, 1, PHYSICAL_VERIFY_TOKENS, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
        )
        self._verify_taps = tuple(
            self._to_tt(torch.zeros(1, 1, PHYSICAL_VERIFY_TOKENS, self.model.hidden_size, dtype=torch.bfloat16))
            for _ in self.dflash_config.target_layer_ids
        )
        self._decode_trace_ids: tuple[int, ...] = ()
        self._fixed_anchor_length = 0
        self._fixed_anchor_start = 0
        self._decode_absolute_position = 0
        self.cached_token_ids: list[int] = []
        self.cached_next_token: int | None = None

    def _to_tt(self, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(tensor, device=self.mesh_device, dtype=dtype, layout=layout)

    def _copy_to_device(self, destination, tensor, *, dtype, layout):
        host = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
        ttnn.copy_host_to_device_tensor(host, destination)

    def _decode_copy(self, tensor, *, memory_config=None, output_tensor=None):
        if self.decode_core_config is not None:
            return self.decode_core_config.copy(
                tensor,
                memory_config=memory_config,
                output_tensor=output_tensor,
            )
        if output_tensor is not None:
            return ttnn.assign(tensor, output_tensor)
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
            return ttnn.clone(tensor, memory_config=memory_config)
        return ttnn.typecast(tensor, tensor.dtype, memory_config=memory_config)

    def _reset_session_cache(self):
        self.model.reset_prefill_state()
        self.drafter.clear_anchor_cache(self.anchor_cache)
        self.anchor_cache = self.drafter.init_anchor_cache()
        self.cached_token_ids.clear()
        self.cached_next_token = None

    def _tokenize(
        self,
        prompt: str | list[dict],
        *,
        tools: list[dict] | None = None,
        reasoning_strength: str = "low",
        tool_namespace_descriptions: dict[str, str] | None = None,
        current_date: str | None = None,
        knowledge_cutoff: str | None = None,
    ) -> torch.Tensor:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        messages = normalize_messages(messages)
        validate_reasoning_strength(reasoning_strength)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            tools=tools,
            reasoning_strength=reasoning_strength,
            tool_namespace_descriptions=tool_namespace_descriptions or {},
            current_date=current_date,
            knowledge_cutoff=knowledge_cutoff,
        )

    def _normalized_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.raw_token_embeddings(token_ids).float()
        embeddings *= torch.rsqrt(embeddings.square().mean(dim=-1, keepdim=True) + self.model.embed_norm_eps)
        return embeddings.to(torch.bfloat16)

    def _ar_embedding(self, indices, table):
        """Keep small AR lookup/tilize programs on the compute subdevice."""
        if self.decode_core_config is None:
            return ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT)
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, int(table.shape[-1])),
            core_grid=one_core,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sharded = ttnn.embedding(
            indices,
            table,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=output_memory_config,
        )
        sharded_4d = ttnn.unsqueeze_to_4D(sharded)
        tiled = ttnn.to_layout(sharded_4d, ttnn.TILE_LAYOUT)
        interleaved = ttnn.sharded_to_interleaved(tiled, ttnn.L1_MEMORY_CONFIG)
        result = ttnn.slice(
            interleaved,
            [0, 0, 0, 0],
            [1, 1, int(indices.shape[-1]), int(table.shape[-1])],
            sub_core_grids=one_core,
        )
        ttnn.deallocate(sharded)
        ttnn.deallocate(tiled)
        ttnn.deallocate(interleaved)
        return result

    def _packed_verify_inputs(self, token_ids: torch.Tensor, current_position: int):
        real_p = int(token_ids.numel())
        physical_p = PHYSICAL_VERIFY_TOKENS
        if real_p > physical_p:
            raise ValueError(f"Packed verify supports at most {physical_p} tokens")

        embeddings = self._normalized_embeddings(token_ids.reshape(1, real_p))
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, physical_p - real_p))
        hidden = self._to_tt(embeddings.unsqueeze(0))

        positions = torch.zeros(1, physical_p, dtype=torch.int32)
        positions[0, :real_p] = torch.arange(current_position, current_position + real_p, dtype=torch.int32)
        position_idx = self._to_tt(positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_positions = torch.full((physical_p,), -1, dtype=torch.int32)
        cur_positions[:real_p] = torch.arange(current_position, current_position + real_p, dtype=torch.int32)
        cur_pos = self._to_tt(cur_positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_index = current_position // KV_PAGE_SIZE
        page_offset = current_position % KV_PAGE_SIZE

        rope_packed = {}
        for layer_type, (cos_cache, sin_cache) in self.model.rope_caches_2d.items():
            cos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
            sin = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
            rope_packed[layer_type] = (cos, sin)

        packed = {
            "p": physical_p,
            "real_p": real_p,
            "position_idx": position_idx,
            "cur_pos": cur_pos,
            "page_index": page_index,
            "page_offset": page_offset,
            "rope_packed": rope_packed,
            "page_table": self.decode_page_table,
        }
        return hidden, packed

    @staticmethod
    def _slice_and_concat_taps(taps, count: int):
        sliced = [ttnn.slice(tap, [0, 0, 0, 0], [1, 1, count, int(tap.shape[-1])]) for tap in taps]
        combined = ttnn.concat(sliced, dim=3)
        for tensor in taps + sliced:
            ttnn.deallocate(tensor)
        return combined

    @staticmethod
    def _top1_ids(logits_tt) -> torch.Tensor:
        """Reduce vocabulary logits on device and transfer only token IDs."""
        logits_rm = ttnn.untilize(logits_tt, use_multicore=True)
        ttnn.deallocate(logits_tt)
        token_ids = ttnn.argmax(logits_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.deallocate(logits_rm)
        token_ids_torch = ttnn.to_torch(ttnn.get_device_tensors(token_ids)[0])
        ttnn.deallocate(token_ids)
        return token_ids_torch[0, 0, :, 0].to(torch.int64)

    @classmethod
    def _top1_at(cls, logits_tt, row: int) -> int:
        return int(cls._top1_ids(logits_tt)[row])

    @staticmethod
    def _release_packed_inputs(packed):
        for name in ("position_idx", "cur_pos"):
            ttnn.deallocate(packed[name])
        for cos, sin in packed["rope_packed"].values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)

    def _prepare_fixed_decode(self, max_new_tokens: int):
        self._fixed_anchor_length, self._fixed_anchor_start = self.drafter.prepare_fixed_anchor_caches(
            self.anchor_cache,
            self.fixed_anchor_caches,
            max_new_tokens=max_new_tokens,
        )
        self._decode_absolute_position = self.anchor_cache.length
        positions = torch.arange(
            self._fixed_anchor_start,
            self._fixed_anchor_start + DFLASH_CACHE_CAPACITY,
            dtype=torch.int64,
        ).unsqueeze(0)
        cos, sin = build_rope_cache(positions, self.dflash_config.head_dim, self.dflash_config.rope_theta)
        self._copy_to_device(
            self._draft_fixed_cos, cos.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._copy_to_device(
            self._draft_fixed_sin, sin.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.drafter.rotate_fixed_anchor_caches(self.fixed_anchor_caches, self._draft_fixed_cos, self._draft_fixed_sin)
        ttnn.synchronize_device(self.mesh_device)

    def _refresh_fixed_draft(self, bonus: int):
        noise_ids = torch.tensor(
            [[bonus] + [self.dflash_config.mask_token_id] * (self.dflash_config.block_size - 1)],
            dtype=torch.long,
        )
        noise = self.model.raw_token_embeddings(noise_ids).unsqueeze(0)
        self._copy_to_device(self._draft_noise, noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        positions = torch.arange(
            self._decode_absolute_position,
            self._decode_absolute_position + self.dflash_config.block_size,
            dtype=torch.int32,
        ).reshape(1, -1)
        self._copy_to_device(self._draft_position_idx, positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._copy_to_device(
            self._draft_valid_len,
            torch.tensor([[self._fixed_anchor_length + self.dflash_config.block_size]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        write_control = torch.zeros(1, 32, dtype=torch.int32)
        write_control[0, 0] = self._fixed_anchor_length
        write_control[0, 1] = self.dflash_config.block_size
        self._copy_to_device(
            self._draft_write_control,
            write_control,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _fixed_draft_forward(self):
        rope_cos, rope_sin = self._ar_rope_caches_2d["sliding_attention"]
        cos_q = ttnn.unsqueeze_to_4D(self._ar_embedding(self._draft_position_idx, rope_cos))
        sin_q = ttnn.unsqueeze_to_4D(self._ar_embedding(self._draft_position_idx, rope_sin))
        mask = self._ar_embedding(self._draft_valid_len, self._draft_mask_table)
        mask = ttnn.reshape(mask, (1, 1, 1, DFLASH_CACHE_CAPACITY))
        mask = ttnn.repeat(
            mask,
            [1, 1, self.dflash_config.num_attention_heads * self.dflash_config.block_size, 1],
            sub_core_grids=self._ar_cores,
        )
        logits = self.drafter.fixed_propose_forward(
            self._draft_noise,
            self.fixed_anchor_caches,
            self._draft_write_control,
            cos_q,
            sin_q,
            mask,
            decode_core_config=self.decode_core_config,
        )
        for tensor in (cos_q, sin_q, mask):
            ttnn.deallocate(tensor)
        return logits

    def _draft_trace_forward(self):
        """DFlash proposal and greedy sampling, entirely on device."""
        logits = self._fixed_draft_forward()
        logits_rm = ttnn.untilize(logits, use_multicore=True, sub_core_grids=self._ar_cores)
        ttnn.deallocate(logits)
        ids = ttnn.argmax(
            logits_rm,
            dim=-1,
            keepdim=True,
            use_multicore=True,
            sub_core_grids=self._ar_cores,
        )
        ttnn.deallocate(logits_rm)
        ids_tiled = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, sub_core_grids=self._ar_cores)
        ttnn.deallocate(ids)
        self._decode_copy(ids_tiled, output_tensor=self._draft_ids)
        ttnn.deallocate(ids_tiled)

    def _refresh_verify(self, token_ids: torch.Tensor):
        real_p = self.dflash_config.block_size
        if int(token_ids.numel()) != real_p:
            raise ValueError(f"Traced verify requires exactly {real_p} input tokens")
        embeddings = self._normalized_embeddings(token_ids.reshape(1, real_p))
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, PHYSICAL_VERIFY_TOKENS - real_p))
        self._copy_to_device(
            self._verify_hidden,
            embeddings.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        positions = torch.zeros(1, PHYSICAL_VERIFY_TOKENS, dtype=torch.int32)
        positions[0, :real_p] = torch.arange(
            self._decode_absolute_position,
            self._decode_absolute_position + real_p,
            dtype=torch.int32,
        )
        self._copy_to_device(
            self._verify_position_idx,
            positions,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._copy_to_device(
            self._append_position_idx,
            positions[:, :real_p],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for layer_type, (cos_cache, sin_cache) in self.model.rope_caches_cpu.items():
            position_rows = positions[0].to(torch.long)
            cos = cos_cache.index_select(0, position_rows).reshape(1, PHYSICAL_VERIFY_TOKENS, 1, -1)
            sin = sin_cache.index_select(0, position_rows).reshape(1, PHYSICAL_VERIFY_TOKENS, 1, -1)
            cos_destination, sin_destination = self._verify_rope[layer_type]
            self._copy_to_device(cos_destination, cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self._copy_to_device(sin_destination, sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cur_pos = torch.full((PHYSICAL_VERIFY_TOKENS,), -1, dtype=torch.int32)
        cur_pos[:real_p] = positions[0, :real_p]
        self._copy_to_device(
            self._verify_cur_pos,
            cur_pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _verify_trace_forward(self):
        def sync_post(stage):
            if self.decode_core_config is not None and os.getenv("MUSE_QKV_POST_DEBUG") == "1":
                ttnn.synchronize_device(
                    self.mesh_device,
                    sub_device_ids=[self.decode_core_config.worker_sub_device_id],
                )
                print(f"MUSE_QKV_POST {stage}", flush=True)

        rope_packed = self._verify_rope if self.decode_core_config is not None else {}
        if self.decode_core_config is None:
            for layer_type, (cos_cache, sin_cache) in self.model.rope_caches_2d.items():
                cos = ttnn.unsqueeze_to_4D(
                    ttnn.embedding(self._verify_position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)
                )
                sin = ttnn.unsqueeze_to_4D(
                    ttnn.embedding(self._verify_position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
                )
                rope_packed[layer_type] = (cos, sin)
        hidden = (
            self.decode_core_config.copy(self._verify_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.decode_core_config is not None
            else ttnn.clone(self._verify_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        logits, _, taps = self.model(
            hidden,
            is_decode=True,
            packed={
                "p": PHYSICAL_VERIFY_TOKENS,
                "real_p": self.dflash_config.block_size,
                "position_idx": self._verify_position_idx,
                "cur_pos": self._verify_cur_pos,
                # Program selection is baked into this request's trace. Use
                # the prompt boundary so long-context replay gets the 64-core
                # SDPA configuration while short decode keeps its lean setup.
                "page_index": self._decode_absolute_position // KV_PAGE_SIZE,
                "page_offset": self._decode_absolute_position % KV_PAGE_SIZE,
                "rope_packed": rope_packed,
                "page_table": self.decode_page_table,
                "retain_tail": False,
            },
            return_aux_hidden=True,
        )
        sync_post("model")
        logits_rm = ttnn.untilize(
            logits,
            use_multicore=True,
            sub_core_grids=self.decode_core_config.compute_cores if self.decode_core_config is not None else None,
        )
        sync_post("untilize")
        ttnn.deallocate(logits)
        ids = ttnn.argmax(
            logits_rm,
            dim=-1,
            keepdim=True,
            use_multicore=True,
            sub_core_grids=self.decode_core_config.argmax_cores if self.decode_core_config is not None else None,
        )
        sync_post("argmax")
        ttnn.deallocate(logits_rm)
        if self.decode_core_config is not None:
            ids_tiled = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, sub_core_grids=self._ar_cores)
            ttnn.deallocate(ids)
            self.decode_core_config.target_copy(ids_tiled, output_tensor=self._verify_ids)
            ttnn.deallocate(ids_tiled)
        else:
            ids_tiled = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
            ttnn.deallocate(ids)
            ttnn.assign(ids_tiled, self._verify_ids)
            ttnn.deallocate(ids_tiled)
        sync_post("ids_copy")
        for tap, destination in zip(taps, self._verify_taps):
            if self.decode_core_config is not None:
                self.decode_core_config.target_copy(tap, output_tensor=destination)
            else:
                ttnn.assign(tap, destination)
            ttnn.deallocate(tap)
        sync_post("tap_copies")
        if self.decode_core_config is None:
            for cos, sin in rope_packed.values():
                ttnn.deallocate(cos)
                ttnn.deallocate(sin)

    def _append_trace_forward(self):
        # Preserve the logical 16-row extent before concatenation.  Concating
        # the 32-row physical padding selects a substantially larger CB plan
        # which overlaps the resident prefetch GCB on the receiver cores.
        tap_slice_cores = (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
            if self.decode_core_config is not None
            else None
        )
        sliced_taps = [
            ttnn.slice(
                tap,
                [0, 0, 0, 0],
                [1, 1, self.dflash_config.block_size, int(tap.shape[-1])],
                sub_core_grids=tap_slice_cores,
            )
            for tap in self._verify_taps
        ]
        if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
            print("MUSE_APPEND tap_slices_start", flush=True)
            ttnn.synchronize_device(self.mesh_device)
            print("MUSE_APPEND tap_slices_end", flush=True)
        combined = ttnn.concat(sliced_taps, dim=3, sub_core_grids=self._ar_cores)
        for tap in sliced_taps:
            ttnn.deallocate(tap)
        if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
            print("MUSE_APPEND taps_concat_start", flush=True)
            ttnn.synchronize_device(self.mesh_device)
            print("MUSE_APPEND taps_concat_end", flush=True)
        rope_cos, rope_sin = self._ar_rope_caches_2d["sliding_attention"]
        cos = ttnn.unsqueeze_to_4D(self._ar_embedding(self._append_position_idx, rope_cos))
        sin = ttnn.unsqueeze_to_4D(self._ar_embedding(self._append_position_idx, rope_sin))
        if os.getenv("MUSE_APPEND_SYNC_ENABLED") == "1":
            print("MUSE_APPEND rope_lookup_start", flush=True)
            ttnn.synchronize_device(self.mesh_device)
            print("MUSE_APPEND rope_lookup_end", flush=True)
        self.drafter.fixed_append_forward(
            self.fixed_anchor_caches,
            combined,
            self._anchor_write_control,
            cos,
            sin,
            decode_core_config=self.decode_core_config,
        )
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(combined)

    def _capture_decode_traces(self):
        """Compile, capture, and retain the three batch-1 AR replay graphs."""

        def trace_debug(stage):
            if os.getenv("MUSE_TRACE_DEBUG") == "1":
                print(f"MUSE_TRACE {stage}", flush=True)

        if self._decode_trace_ids:
            raise RuntimeError("Decode traces are already active")
        for destination in (self._draft_write_control, self._anchor_write_control):
            self._copy_to_device(
                destination,
                torch.zeros(1, 32, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        empty_verify = torch.zeros(self.dflash_config.block_size, dtype=torch.long)
        self._refresh_verify(empty_verify)
        ttnn.synchronize_device(self.mesh_device)
        forwards = (self._draft_trace_forward, self._verify_trace_forward, self._append_trace_forward)
        if self.qkv_prefetcher is None:
            for forward in forwards:
                forward()
                ttnn.synchronize_device(self.mesh_device)
            trace_ids = []
            for forward in forwards:
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    forward()
                except Exception:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.release_trace(self.mesh_device, trace_id)
                    for captured_id in trace_ids:
                        ttnn.release_trace(self.mesh_device, captured_id)
                    raise
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                trace_ids.append(trace_id)
        else:
            # Install the decode core policy before compiling any AR program so
            # the program cache never aliases an all-grid warmup with a
            # worker-only capture.  The receiver ring and its control block
            # remain reserved for the complete decode lifetime.
            self._activate_qkv_prefetcher()
            self.mesh_device.reset_sub_device_stall_group()
            for index in (0, 2):
                trace_debug(f"warmup_{index}_start")
                overlay_env = os.environ.get("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY")
                os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = "1"
                try:
                    forwards[index]()
                    ttnn.synchronize_device(self.mesh_device)
                finally:
                    if overlay_env is None:
                        os.environ.pop("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY", None)
                    else:
                        os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = overlay_env
                trace_debug(f"warmup_{index}_end")

            trace_debug("verify_warmup_start")
            self._run_qkv_prefetcher()
            self._verify_trace_forward()
            self.mesh_device.reset_sub_device_stall_group()
            ttnn.synchronize_device(self.mesh_device)
            trace_debug("verify_warmup_end")

            trace_ids = []
            for index, forward in enumerate(forwards):
                trace_debug(f"capture_{index}_start")
                if index == 1:
                    self._activate_qkv_prefetcher()
                if index != 1:
                    # Target warmup replaces kernels on the shared compute
                    # rectangle. Reload this phase's binaries immediately
                    # before capture.
                    overlay_env = os.environ.get("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY")
                    os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = "1"
                    try:
                        forward()
                        ttnn.synchronize_device(self.mesh_device)
                    finally:
                        if overlay_env is None:
                            os.environ.pop("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY", None)
                        else:
                            os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = overlay_env
                else:
                    # Likewise, draft capture replaces target kernels. Consume
                    # a fresh stream to reload target programs, then re-arm the
                    # stall group for the stream recorded in the trace.
                    self._run_qkv_prefetcher()
                    forward()
                    self.mesh_device.reset_sub_device_stall_group()
                    ttnn.synchronize_device(self.mesh_device)
                    self._activate_qkv_prefetcher()
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                overlay_env = os.environ.get("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY")
                if index != 1:
                    os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = "1"
                try:
                    if index == 1:
                        # Capture the producer launch with its consumers. Fast
                        # dispatch still runs the sender and worker sub-devices
                        # concurrently, while one CQ gives replay a single
                        # unambiguous ownership order.
                        self._run_qkv_prefetcher(cq_id=0)
                    forward()
                except Exception:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.release_trace(self.mesh_device, trace_id)
                    for captured_id in trace_ids:
                        ttnn.release_trace(self.mesh_device, captured_id)
                    raise
                finally:
                    if index != 1:
                        if overlay_env is None:
                            os.environ.pop("TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY", None)
                        else:
                            os.environ["TT_METAL_ALLOW_L1_TEMPORAL_OVERLAY"] = overlay_env
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                trace_ids.append(trace_id)
                if index == 1:
                    ttnn.synchronize_device(self.mesh_device)
                    self.mesh_device.reset_sub_device_stall_group()
                else:
                    ttnn.synchronize_device(self.mesh_device)
                trace_debug(f"capture_{index}_end")
        ttnn.synchronize_device(self.mesh_device)
        self._decode_trace_ids = tuple(trace_ids)

    def _activate_qkv_prefetcher(self):
        if not self.qkv_prefetcher.init_decode_done:
            self.qkv_prefetcher.init(Mode.DECODE)
            for layer in self.model.layers[: self.qkv_prefetcher.num_layers]:
                weights = layer.self_attn.weights.wqkv_prefetch
                if weights is None:
                    raise RuntimeError("MUSE_QKV_PREFETCH requires DRAM-sharded QKV weights")
                for weight in weights:
                    self.qkv_prefetcher.insert_tensor(weight)
                if self._prefetch_mlp:
                    for weights in (
                        layer.mlp.gate_proj_prefetch,
                        layer.mlp.up_proj_prefetch,
                    ):
                        if weights is None:
                            raise RuntimeError("MUSE_PREFETCH_MLP requires DRAM-sharded MLP gate/up weights")
                        for weight in weights:
                            self.qkv_prefetcher.insert_tensor(weight)
            self.decode_core_config = MuseDecodeCoreConfig(
                self.qkv_prefetcher,
            )
            self.model.configure_decode_cores(self.decode_core_config)
            # Reserve ring data and control storage before any draft, verify,
            # or append program is compiled.  This prevents a cached program
            # from silently reusing the persistent prefetch L1 region.
            self.qkv_prefetcher.prepare_global_cb()
        else:
            manager = self.qkv_prefetcher.prefetcher_sub_device.manager_id
            self.mesh_device.load_sub_device_manager(manager)
            # A captured verify trace begins with the prefetch program launch.
            # Keep only workers stalled so the dedicated sender sub-device can
            # consume that command and start filling the GCB.  Stalling both
            # sub-devices here deadlocks replay before the trace's captured
            # stall-group transition can execute.
        self.mesh_device.set_sub_device_stall_group(self.qkv_prefetcher.consumer_sub_device_ids)

    def _run_qkv_prefetcher(self, cq_id=0):
        # Sender and worker kernels occupy separate sub-devices, so fast
        # dispatch overlaps them even when their launches share one CQ.
        with ttnn.command_queue(cq_id):
            self.qkv_prefetcher.run()

    def _deactivate_qkv_prefetcher(self):
        # Verify's ID readback can complete before the producer and the final
        # tap copies have retired.  Draft/append deliberately reuse the
        # receiver cores' L1 after the stream, so establish a full phase
        # boundary before allowing either graph to replace their CB state.
        # Producer and consumers are in the same trace/CQ. Retire both before
        # draft/append reuse the receiver ring's L1 region.
        ttnn.synchronize_device(
            self.mesh_device,
            cq_id=0,
            sub_device_ids=self.qkv_prefetcher.consumer_sub_device_ids
            + self.qkv_prefetcher.prefetcher_sub_device.sub_devices_id,
        )
        # Readback and the draft/append traces run on the ordinary compute
        # sub-device.  Once the captured verify work has retired, release the
        # consumer stall group so those subsequent CQ0 commands can dispatch.
        self.mesh_device.reset_sub_device_stall_group()

    def _release_decode_traces(self):
        for trace_id in self._decode_trace_ids:
            ttnn.release_trace(self.mesh_device, trace_id)
        self._decode_trace_ids = ()

    def _read_ids(self, tensor, count: int) -> list[int]:
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
        return host.reshape(-1)[:count].to(torch.int64).tolist()

    def _append_fixed_anchors(self, committed_count: int):
        write_control = torch.zeros(1, 32, dtype=torch.int32)
        write_control[0, 0] = self._fixed_anchor_length
        write_control[0, 1] = committed_count
        self._copy_to_device(
            self._anchor_write_control,
            write_control,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if os.getenv("MUSE_APPEND_EAGER_DEBUG") == "1":
            if os.getenv("MUSE_APPEND_SYNC_DEBUG") == "1":
                os.environ["MUSE_APPEND_SYNC_ENABLED"] = "1"
            try:
                self._append_trace_forward()
            finally:
                os.environ.pop("MUSE_APPEND_SYNC_ENABLED", None)
        else:
            ttnn.execute_trace(self.mesh_device, self._decode_trace_ids[2], cq_id=0, blocking=False)
        self._fixed_anchor_length += committed_count
        self._decode_absolute_position += committed_count

    def _append_prompt_tokens(self, token_ids: list[int]) -> tuple[int, int, int]:
        """Append uncached tokens; return next token plus chunked/packed counts."""
        offset = 0
        next_token = self.cached_next_token
        chunked_tokens = 0
        packed_tokens = 0
        while offset < len(token_ids):
            position = len(self.cached_token_ids)
            remaining = len(token_ids) - offset
            if position % PREFILL_ALIGNMENT == 0 and remaining >= PREFILL_ALIGNMENT:
                chunk_len = min(PREFILL_CHUNK_SIZE, (remaining // PREFILL_ALIGNMENT) * PREFILL_ALIGNMENT)
                _profile_signpost("MUSE_CHUNKED_PREFILL_START", f"position={position},tokens={chunk_len}")
                processed = token_ids[offset : offset + chunk_len]
                chunk_ids = torch.tensor(processed, dtype=torch.long).reshape(1, -1)
                needs_next_token = offset + chunk_len == len(token_ids)
                logits_tt, _, taps = self.model(
                    self.model.embed_input_ids(chunk_ids),
                    is_decode=False,
                    packed={
                        "page_table": self.model.page_table,
                        "chunk_start": position,
                        "logical_length": chunk_len,
                    },
                    return_aux_hidden=True,
                    last_token_only=needs_next_token,
                    compute_logits=needs_next_token,
                )
                prompt_aux = self._slice_and_concat_taps(taps, chunk_len)
                self.drafter.append_anchors(self.anchor_cache, prompt_aux)
                ttnn.deallocate(prompt_aux)
                if needs_next_token:
                    next_token = self._top1_at(logits_tt, 0)
                chunked_tokens += chunk_len
                _profile_signpost("MUSE_CHUNKED_PREFILL_END", f"position={position},tokens={chunk_len}")
            else:
                until_aligned = PREFILL_ALIGNMENT - (position % PREFILL_ALIGNMENT)
                use_wide_pack = (position % PREFILL_ALIGNMENT != 0 and remaining >= until_aligned) or (
                    position >= PREFILL_CHUNK_SIZE and remaining <= PHYSICAL_VERIFY_TOKENS
                )
                packed_capacity = PHYSICAL_VERIFY_TOKENS if use_wide_pack else self.dflash_config.block_size
                append_len = min(packed_capacity, remaining, until_aligned)
                _profile_signpost("MUSE_PACKED_PREFILL_START", f"position={position},tokens={append_len}")
                processed = token_ids[offset : offset + append_len]
                chunk_ids = torch.tensor(processed, dtype=torch.long)
                hidden, packed = self._packed_verify_inputs(chunk_ids, position)
                needs_next_token = offset + append_len == len(token_ids)
                logits_tt, _, taps = self.model(
                    hidden,
                    is_decode=True,
                    packed=packed,
                    return_aux_hidden=True,
                    last_token_only=needs_next_token,
                    compute_logits=needs_next_token,
                )
                self.model.commit_decode_tails(append_len)
                prompt_aux = self._slice_and_concat_taps(taps, append_len)
                self.drafter.append_anchors(self.anchor_cache, prompt_aux)
                ttnn.deallocate(prompt_aux)
                if needs_next_token:
                    next_token = self._top1_at(logits_tt, 0)
                self._release_packed_inputs(packed)
                packed_tokens += append_len
                _profile_signpost("MUSE_PACKED_PREFILL_END", f"position={position},tokens={append_len}")

            self.cached_token_ids.extend(processed)
            offset += len(processed)

        if self.anchor_cache.length != len(self.cached_token_ids):
            raise RuntimeError("Target and DFlash cache positions diverged during prefill")
        if next_token is None:
            raise RuntimeError("Cannot generate from an empty prompt cache")
        self.cached_next_token = next_token
        return next_token, chunked_tokens, packed_tokens

    def _speculative_decode(
        self,
        bonus: int,
        max_new_tokens: int,
        on_token_ids: Callable[[list[int]], None] | None = None,
    ):
        generated = [bonus]
        if on_token_ids:
            on_token_ids(list(generated))
        accepted_counts = []
        eos_ids = self.eos_token_ids
        profile_breakdown = os.getenv("MUSE_DECODE_BREAKDOWN") == "1"
        breakdown = {"draft": 0.0, "verify": 0.0, "readback": 0.0, "commit": 0.0}
        if len(self._decode_trace_ids) != 3:
            raise RuntimeError("AR decode requires captured draft, verify, and append traces")

        def stage_start():
            if not profile_breakdown:
                return 0.0
            ttnn.synchronize_device(self.mesh_device)
            return time.perf_counter()

        def stage_end(name, started):
            if profile_breakdown:
                ttnn.synchronize_device(self.mesh_device)
                breakdown[name] += time.perf_counter() - started

        while len(generated) < max_new_tokens and generated[-1] not in eos_ids:
            trace_runtime_debug = os.getenv("MUSE_TRACE_DEBUG") == "1"
            draft_started = stage_start()
            if os.getenv("MUSE_VERIFY_FIRST_DEBUG") == "1" and len(generated) == 1:
                drafts = [0] * (self.dflash_config.block_size - 1)
            else:
                if trace_runtime_debug:
                    print("MUSE_TRACE replay_draft_start", flush=True)
                _profile_signpost("MUSE_DFLASH_DRAFT_START", f"position={self._decode_absolute_position}")
                self._refresh_fixed_draft(bonus)
                if trace_runtime_debug:
                    print("MUSE_TRACE replay_draft_refresh_end", flush=True)
                ttnn.execute_trace(self.mesh_device, self._decode_trace_ids[0], cq_id=0, blocking=False)
                if trace_runtime_debug:
                    print("MUSE_TRACE replay_draft_execute_end", flush=True)
                drafts = self._read_ids(self._draft_ids, self.dflash_config.block_size - 1)
                if trace_runtime_debug:
                    print("MUSE_TRACE replay_draft_end", flush=True)
            stage_end("draft", draft_started)
            _profile_signpost("MUSE_DFLASH_DRAFT_END")

            verify_started = stage_start()
            _profile_signpost("MUSE_TARGET_VERIFY_START", f"position={self._decode_absolute_position}")
            verify_ids = torch.tensor([bonus] + drafts, dtype=torch.long)
            self._refresh_verify(verify_ids)
            if self.qkv_prefetcher is not None:
                # Draft/append time-multiplex the empty receiver data ring.
                # Their result may become readable before all tail kernels
                # have retired, so establish the phase boundary before refill.
                ttnn.synchronize_device(
                    self.mesh_device,
                    cq_id=0,
                    sub_device_ids=self.qkv_prefetcher.consumer_sub_device_ids,
                )
                self._activate_qkv_prefetcher()
            if trace_runtime_debug:
                print("MUSE_TRACE replay_verify_start", flush=True)
            if os.getenv("MUSE_VERIFY_EAGER_DEBUG") == "1":
                self._verify_trace_forward()
            else:
                ttnn.execute_trace(self.mesh_device, self._decode_trace_ids[1], cq_id=0, blocking=False)
            stage_end("verify", verify_started)
            if trace_runtime_debug:
                print("MUSE_TRACE replay_verify_end", flush=True)
            _profile_signpost("MUSE_TARGET_VERIFY_END")

            readback_started = stage_start()
            if self.qkv_prefetcher is not None:
                self._deactivate_qkv_prefetcher()
            target_top1 = self._read_ids(self._verify_ids, self.dflash_config.block_size)
            if trace_runtime_debug:
                print("MUSE_TRACE replay_readback_end", flush=True)
            stage_end("readback", readback_started)

            matches = 0
            while matches < len(drafts) and drafts[matches] == target_top1[matches]:
                matches += 1
            accepted_counts.append(matches)

            room = max_new_tokens - len(generated)
            accepted_count = min(matches, room)
            for index, token in enumerate(drafts[:accepted_count]):
                if token in eos_ids:
                    accepted_count = index + 1
                    break
            committed_inputs = accepted_count + 1  # previous bonus plus returned accepted drafts
            commit_started = stage_start()
            if trace_runtime_debug:
                print("MUSE_TRACE replay_commit_start", flush=True)
            self._append_fixed_anchors(committed_inputs)
            if trace_runtime_debug:
                print("MUSE_TRACE replay_commit_end", flush=True)
            stage_end("commit", commit_started)
            committed_ids = [bonus] + drafts[:accepted_count]
            self.cached_token_ids.extend(committed_ids)
            self.cached_next_token = int(target_top1[committed_inputs - 1])

            accepted = drafts[:accepted_count]
            generated.extend(accepted)
            if len(generated) >= max_new_tokens or any(token in eos_ids for token in accepted):
                if on_token_ids:
                    on_token_ids(list(generated[:max_new_tokens]))
                break
            bonus = int(target_top1[matches])
            generated.append(bonus)
            if on_token_ids:
                on_token_ids(list(generated))

        generated = generated[:max_new_tokens]
        for index, token in enumerate(generated):
            if token in eos_ids:
                generated = generated[: index + 1]
                break
        if profile_breakdown:
            cycles = len(accepted_counts)
            total = sum(breakdown.values())
            print(
                "decode breakdown: "
                + "; ".join(f"{name}={seconds:.6f}s" for name, seconds in breakdown.items())
                + f"; total={total:.6f}s; cycles={cycles}; generated={len(generated)}; "
                + f"accepted={sum(accepted_counts)}",
                flush=True,
            )
        return generated, accepted_counts

    def generate(
        self,
        prompt: str | list[dict],
        max_new_tokens: int = 32,
        *,
        tools: list[dict] | None = None,
        reasoning_strength: str = "low",
        tool_namespace_descriptions: dict[str, str] | None = None,
        current_date: str | None = None,
        knowledge_cutoff: str | None = None,
        on_token_ids: Callable[[list[int]], None] | None = None,
    ) -> GenerationResult:
        if max_new_tokens < 1:
            return GenerationResult("", [], [], 0.0, finish_reason="length")
        started = time.perf_counter()
        _profile_signpost("MUSE_REQUEST_START")
        tokenization_started = time.perf_counter()
        input_ids = self._tokenize(
            prompt,
            tools=tools,
            reasoning_strength=reasoning_strength,
            tool_namespace_descriptions=tool_namespace_descriptions,
            current_date=current_date,
            knowledge_cutoff=knowledge_cutoff,
        )
        tokenization_seconds = time.perf_counter() - tokenization_started
        prompt_tokens = [int(token) for token in input_ids[0].tolist()]
        prompt_len = len(prompt_tokens)
        if prompt_len + max_new_tokens + self.dflash_config.block_size > self.max_seq_len:
            raise ValueError("Prompt plus generation exceeds max_seq_len")
        if max_new_tokens > DFLASH_CACHE_CAPACITY - self.dflash_config.block_size:
            raise ValueError(
                f"One DFlash completion supports at most {DFLASH_CACHE_CAPACITY - self.dflash_config.block_size} tokens"
            )
        prefix_reused = len(self.cached_token_ids)
        if prompt_tokens[:prefix_reused] != self.cached_token_ids:
            self._reset_session_cache()
            prefix_reused = 0
        suffix = prompt_tokens[prefix_reused:]
        _profile_signpost("MUSE_PREFILL_START", f"cached={prefix_reused},uncached={len(suffix)},total={prompt_len}")
        prefill_started = time.perf_counter()
        if suffix:
            bonus, chunked_prefill_tokens, packed_prefill_tokens = self._append_prompt_tokens(suffix)
        else:
            bonus = self.cached_next_token
            chunked_prefill_tokens = 0
            packed_prefill_tokens = 0
        if suffix:
            ttnn.synchronize_device(self.mesh_device)
        prefill_seconds = time.perf_counter() - prefill_started if suffix else 0.0
        _profile_signpost("MUSE_PREFILL_END", f"seconds={prefill_seconds:.6f}")
        if bonus is None:
            raise RuntimeError("Cached prompt is missing its next-token prediction")
        # Decode mutates sliding tails and DFlash anchors.  Keep an owning
        # prompt-boundary snapshot, then restore it so a following chat turn
        # can reuse the exact input prefix even when the tokenizer normalizes
        # the prior generated assistant/tool message differently.
        snapshot_started = time.perf_counter()
        _profile_signpost("MUSE_SNAPSHOT_START")
        target_prompt_state = self.model.clone_prefill_state()
        dflash_prompt_state = self.drafter.clone_anchor_cache(self.anchor_cache)
        cache_snapshot_seconds = 0.0
        decode_seconds = 0.0
        try:
            self._prepare_fixed_decode(max_new_tokens)
            self._capture_decode_traces()
            ttnn.synchronize_device(self.mesh_device)
            cache_snapshot_seconds = time.perf_counter() - snapshot_started
            _profile_signpost("MUSE_SNAPSHOT_END", f"seconds={cache_snapshot_seconds:.6f}")
            decode_started = time.perf_counter()
            _profile_signpost("MUSE_DECODE_START")
            generated, accepted_counts = self._speculative_decode(
                bonus,
                max_new_tokens,
                on_token_ids=on_token_ids,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_seconds = time.perf_counter() - decode_started
            _profile_signpost("MUSE_DECODE_END", f"seconds={decode_seconds:.6f},tokens={len(generated)}")
        finally:
            if self._decode_trace_ids:
                self._release_decode_traces()
            if self.qkv_prefetcher is not None and self.qkv_prefetcher.init_decode_done:
                self.mesh_device.reset_sub_device_stall_group()
                self.mesh_device.clear_loaded_sub_device_manager()
            restore_started = time.perf_counter()
            self.model.restore_prefill_state(target_prompt_state)
            self.drafter.clear_anchor_cache(self.anchor_cache)
            self.anchor_cache = dflash_prompt_state
            self.cached_token_ids = list(prompt_tokens)
            self.cached_next_token = bonus
            ttnn.synchronize_device(self.mesh_device)
            cache_snapshot_seconds += time.perf_counter() - restore_started

        parsed = parse_tokenizer_response(
            self.tokenizer,
            generated,
            tools=tools,
            reached_stop_token=bool(generated and generated[-1] in self.eos_token_ids),
        )
        _profile_signpost("MUSE_REQUEST_END")
        return GenerationResult(
            text=parsed.content or "",
            token_ids=generated,
            accepted_drafts=accepted_counts,
            elapsed_seconds=time.perf_counter() - started,
            reasoning_content=parsed.reasoning_content,
            tool_calls=parsed.tool_calls,
            recipient=parsed.recipient,
            finish_reason=parsed.finish_reason,
            prompt_tokens=prompt_len,
            cached_prompt_tokens=prefix_reused,
            prefilled_prompt_tokens=len(suffix),
            chunked_prefill_tokens=chunked_prefill_tokens,
            packed_prefill_tokens=packed_prefill_tokens,
            tokenization_seconds=tokenization_seconds,
            prefill_seconds=prefill_seconds,
            cache_snapshot_seconds=cache_snapshot_seconds,
            decode_seconds=decode_seconds,
        )


MODEL_NAME = "meta-models/Muse-Glimmer-30B"


def _usage(result: GenerationResult) -> dict:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": len(result.token_ids),
        "total_tokens": result.prompt_tokens + len(result.token_ids),
        "prompt_tokens_details": {"cached_tokens": result.cached_prompt_tokens},
    }


def _dflash_metrics(result: GenerationResult) -> dict:
    return {
        "accepted_drafts": result.accepted_drafts,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "prefilled_prompt_tokens": result.prefilled_prompt_tokens,
        "chunked_prefill_tokens": result.chunked_prefill_tokens,
        "packed_prefill_tokens": result.packed_prefill_tokens,
        "prefill_seconds": result.prefill_seconds,
        "prefill_tokens_per_second": result.prefill_tokens_per_second,
        "decode_seconds": result.decode_seconds,
        "decode_tokens_per_second": result.decode_tokens_per_second,
        "ar_decode_tokens": result.ar_decode_tokens,
        "ar_decode_tokens_per_second": result.ar_decode_tokens_per_second,
        "tokenization_seconds": result.tokenization_seconds,
        "cache_snapshot_seconds": result.cache_snapshot_seconds,
    }


def build_app(engine: Engine):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="Muse Glimmer DFlash")
    request_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "speculator": "native-dflash"}

    def generate(request: ChatRequest, active_tools, *, on_token_ids=None):
        return engine.generate(
            request.messages,
            request.max_tokens,
            tools=active_tools,
            reasoning_strength=request.reasoning_strength,
            tool_namespace_descriptions=request.tool_namespace_descriptions,
            current_date=request.current_date,
            knowledge_cutoff=request.knowledge_cutoff,
            on_token_ids=on_token_ids,
        )

    def stream_completion(request: ChatRequest, active_tools):
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
        }
        if (request.stream_options or {}).get("include_usage"):
            # OpenAI emits null usage on ordinary chunks and one final
            # choices=[] chunk containing the aggregate counts.
            base["usage"] = None
        output = queue.Queue()

        def run_generation():
            try:
                with request_lock:
                    result = generate(request, active_tools, on_token_ids=lambda ids: output.put(("tokens", ids)))
                output.put(("result", result))
            except Exception as error:  # The HTTP status has already streamed.
                output.put(("error", error))

        threading.Thread(target=run_generation, daemon=True, name="muse-glimmer-stream").start()
        decoder = IncrementalMuseResponse(engine.tokenizer)

        def event(payload):
            return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"

        # OpenAI streams the assistant role once, before text deltas.
        yield event(
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
        )
        while True:
            kind, payload = output.get()
            if kind == "tokens":
                for delta_kind, text in decoder.update(payload):
                    key = "reasoning_content" if delta_kind == "reasoning" else "content"
                    yield event({**base, "choices": [{"index": 0, "delta": {key: text}, "finish_reason": None}]})
                continue
            if kind == "error":
                error_type = "invalid_request_error" if isinstance(payload, ValueError) else "server_error"
                yield event({"error": {"message": str(payload), "type": error_type}})
                yield "data: [DONE]\n\n"
                return

            result = payload
            # Flush any tokenizer-normalized suffix before structured calls.
            for delta_kind, text in decoder.finish(result):
                key = "reasoning_content" if delta_kind == "reasoning" else "content"
                yield event({**base, "choices": [{"index": 0, "delta": {key: text}, "finish_reason": None}]})
            for index, tool_call in enumerate(result.tool_calls or []):
                yield event(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": index, **tool_call}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield event(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}],
                }
            )
            if (request.stream_options or {}).get("include_usage"):
                yield event({**base, "choices": [], "usage": _usage(result), "dflash": _dflash_metrics(result)})
            yield "data: [DONE]\n\n"
            return

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        from fastapi import HTTPException

        try:
            if request.tool_choice not in (None, "auto", "none"):
                raise ValueError("tool_choice supports only 'auto' or 'none'")
            active_tools = None if request.tool_choice == "none" else request.tools
            if request.stream:
                return StreamingResponse(
                    stream_completion(request, active_tools),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            with request_lock:
                result = generate(request, active_tools)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        message = {"role": "assistant", "content": result.text or None}
        if result.reasoning_content:
            message["reasoning_content"] = result.reasoning_content
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        return {
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
            "usage": _usage(result),
            "dflash": _dflash_metrics(result),
        }

    return app


def run_server(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--assistant-path", default=DEFAULT_ASSISTANT_PATH)
    parser.add_argument("--assistant-cache-path")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    opts = parser.parse_args(args)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        trace_region_size=opts.trace_region_size,
        num_command_queues=2 if os.getenv("MUSE_QKV_PREFETCH", "0") == "1" else 1,
    )
    engine = Engine(
        mesh_device,
        opts.model_path,
        opts.assistant_path,
        opts.max_seq_len,
        assistant_cache_path=opts.assistant_cache_path,
    )
    if opts.prompt is not None:
        try:
            result = engine.generate(opts.prompt, opts.max_new_tokens)
            print(result.text, flush=True)
            print(
                f"DFlash accepted drafts per verify: {result.accepted_drafts}; "
                f"elapsed={result.elapsed_seconds:.2f}s; end-to-end tok/s={result.tokens_per_second:.2f}; "
                f"prefill tok/s={result.prefill_tokens_per_second:.2f}; "
                f"AR decode tokens={result.ar_decode_tokens}; "
                f"AR decode tok/s={result.ar_decode_tokens_per_second:.2f}",
                flush=True,
            )
        finally:
            ttnn.close_mesh_device(mesh_device)
        return

    import uvicorn

    try:
        uvicorn.run(build_app(engine), host=opts.host, port=opts.port)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    run_server()
