# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MTP drafter — Gemma 4's official 4-layer assistant model.

Important: this drafter is **NOT a standalone model**. Unlike n-gram or a
classic draft-LM, the Gemma 4 MTP drafter is architecturally coupled to
the target model. Its `forward()` requires:

- ``inputs_embeds``: hidden states from the target model's last decoder
  layer (after `pre_projection` it expects shape [B, T, hidden_target]).
- ``shared_kv_states``: dict mapping layer_type → (K, V) tensors from the
  target's LAST layer of each type (full_attention, sliding_attention).

This is the "MTP head" pattern — the drafter is effectively 4 extra
decoder layers that consume the target's representations + share its KV
cache from the deepest layer. It cannot run on raw token IDs alone.

What this file provides
-----------------------

1. ``MTPDrafter`` class with a ``benchmark_forward()`` method that runs the
   drafter with synthetic ``inputs_embeds`` and ``shared_kv_states`` of the
   shapes the target would produce. Measures wall-clock cost per draft
   token, which is what you need for the speedup math.

2. Documentation of the API contract for whoever wires this into cody's
   TT decode path. The integration is the path forward; until then, this
   wrapper measures cost only.

Why no corpus-based simulation?
-------------------------------

For n-gram the simulation worked because n-gram is independent of the
target model — it just predicts from the request's own history. For MTP,
the drafter's quality is *defined* by how well it mimics the specific
target model it was trained against, and its forward pass requires the
target's hidden states. There's no meaningful way to measure acceptance
rate without running the target alongside.

Acceptance rate per the Gemma 4 paper / model card: claims "up to 3×
speedups" which corresponds to per-token acceptance ~0.70-0.80 against
the matched target. We use this published number for projections.

Integration path for cody (TT hardware)
---------------------------------------

Step 1 (this file): document the API + measure drafter forward cost.
Step 2 (separate work): wire into cody's `_step_decode` so each step:
    a. After the target's full forward pass on token t, extract the last
       layer's hidden state (post pre_attention_layernorm, before lm_head)
       AND the last-layer KV from each layer-type (full / sliding).
    b. Feed those to the drafter's 4-layer forward.
    c. Drafter emits T predictions sequentially (autoregressively over
       the drafter's own outputs, with the target's hidden states only
       fed for the first step).
    d. Stage those T tokens as drafts for the NEXT packed-decode step.
Step 3 (separate work): build the verify + reject + KV rollback logic.

See `README.md` for the install command + verification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetModelShapes:
    """Shapes of the target model state that the drafter consumes.

    For Gemma 4 26B-A4B-it (cody's target):
        backbone_hidden_size = 2816  (residual-stream dim; NOT 5376, which is
            the post-MoE projection output)
        num_kv_heads_sliding = 8, head_dim_sliding = 256
        num_kv_heads_full = 2,    head_dim_full = 512 (Gemma's "global_head_dim")

    The drafter's pre_projection takes `2 * backbone_hidden_size`-wide
    inputs (per the modeling code at modeling_gemma4_assistant.py:125).
    Likely this is [hidden_state, next_token_embed] concatenated; the exact
    semantics live in the target model's integration code.

    These defaults match the assistant config at /mnt/nas/gemma-assistant.
    """

    backbone_hidden: int = 2816
    num_kv_heads_sliding: int = 8
    head_dim_sliding: int = 256
    num_kv_heads_full: int = 2
    head_dim_full: int = 512
    max_seq_len: int = 4096


class MTPDrafter:
    """Gemma 4 MTP drafter wrapper.

    Holds the assistant model and provides a benchmark harness for its
    forward cost. Cannot operate without target model state (see module
    docstring).
    """

    def __init__(
        self,
        assistant_model_path: str,
        device: str = "cpu",
        dtype: str = "float16",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("MTPDrafter requires torch + transformers. " f"Install error: {e}")

        self._torch = torch
        self._dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(assistant_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            assistant_model_path,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = text_cfg.num_hidden_layers
        self.hidden_size = text_cfg.hidden_size
        self.head_dim = text_cfg.head_dim
        self.num_kv_heads = text_cfg.num_key_value_heads

    def stats(self) -> dict:
        return {
            "model_path": getattr(self.model.config, "_name_or_path", "?"),
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "device": str(self.device),
            "dtype": str(self._dtype),
        }

    def benchmark_forward(
        self,
        target_shapes: TargetModelShapes,
        num_steps: int = 50,
        warmup: int = 5,
        batch: int = 1,
        kv_len: int = 128,
    ) -> dict:
        """Time the drafter's forward pass with synthetic target inputs.

        Builds fake ``inputs_embeds`` and ``shared_kv_states`` matching the
        target model's output shape, then runs the drafter ``num_steps``
        times.

        Args:
            target_shapes: TargetModelShapes describing the target's hidden
                size and KV layout. Defaults match Gemma 4 26B-A4B-it.
            num_steps: how many forward passes to time (post-warmup).
            warmup: how many initial passes to discard.
            batch: B in target's hidden states. cody runs at B=32.
            kv_len: length of the shared KV (i.e. how much context the
                target has consumed before drafting starts).

        Returns:
            Dict with mean / min / max wall-clock ms per drafter forward.
        """
        import time

        torch = self._torch

        H = target_shapes.backbone_hidden  # 2816 for 26B-A4B-it
        # pre_projection expects [B, T, 2*H]. T=1 for single-step proposals.
        inputs_embeds = torch.randn(batch, 1, 2 * H, dtype=self._dtype, device=self.device)
        # shared_kv_states: dict[layer_type → (K, V)] with per-type head shapes.
        shared_kv = {
            "full_attention": (
                torch.randn(
                    batch,
                    target_shapes.num_kv_heads_full,
                    kv_len,
                    target_shapes.head_dim_full,
                    dtype=self._dtype,
                    device=self.device,
                ),
                torch.randn(
                    batch,
                    target_shapes.num_kv_heads_full,
                    kv_len,
                    target_shapes.head_dim_full,
                    dtype=self._dtype,
                    device=self.device,
                ),
            ),
            "sliding_attention": (
                torch.randn(
                    batch,
                    target_shapes.num_kv_heads_sliding,
                    kv_len,
                    target_shapes.head_dim_sliding,
                    dtype=self._dtype,
                    device=self.device,
                ),
                torch.randn(
                    batch,
                    target_shapes.num_kv_heads_sliding,
                    kv_len,
                    target_shapes.head_dim_sliding,
                    dtype=self._dtype,
                    device=self.device,
                ),
            ),
        }
        attn_mask = torch.ones(batch, kv_len, dtype=torch.long, device=self.device)

        def one_call():
            with torch.no_grad():
                out = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    shared_kv_states=shared_kv,
                )
            # CPU has no async, but be safe for cuda
            if self.device != "cpu":
                torch.cuda.synchronize()
            return out

        for _ in range(warmup):
            one_call()

        samples = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            one_call()
            samples.append((time.perf_counter() - t0) * 1e3)

        return {
            "mean_ms": sum(samples) / len(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "num_samples": num_steps,
            "shapes": {
                "batch": batch,
                "backbone_hidden": H,
                "kv_len": kv_len,
                "sliding": (target_shapes.num_kv_heads_sliding, target_shapes.head_dim_sliding),
                "full": (target_shapes.num_kv_heads_full, target_shapes.head_dim_full),
            },
            "stats": self.stats(),
        }
