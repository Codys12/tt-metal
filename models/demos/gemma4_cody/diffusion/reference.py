# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pure-torch CPU reference for DiffusionGemma (text-only), vendored from
transformers 5.11.0 `models/diffusion_gemma/modeling_diffusion_gemma.py`.

The installed venv transformers (5.9.0) has no `diffusion_gemma` model type, so
this file reimplements the text encoder/decoder stack standalone — no Cache
class, no masking utils. It exists for PCC parity tests against the TT
implementation; vision is intentionally not supported.

Weight keys (HF checkpoint, all text weights are shared encoder<->decoder
except per-layer `layer_scalar`):
    model.decoder.embed_tokens.weight
    model.decoder.layers.{i}.{...}                 (decoder layer_scalar here)
    model.encoder.language_model.layers.{i}.layer_scalar
    model.decoder.norm.weight
    model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}.weight
LM head is tied to embed_tokens. Logit softcap 30.0, attention scale 1.0.
"""

import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class DiffusionTextConfig:
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    num_global_key_value_heads: int = 2
    global_head_dim: int = 512
    sliding_window: int = 1024
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    canvas_length: int = 256
    layer_types: tuple = ()
    rope_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, model_path):
        with open(f"{model_path}/config.json") as f:
            cfg = json.load(f)
        tc = cfg["text_config"]
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            num_global_key_value_heads=tc["num_global_key_value_heads"],
            global_head_dim=tc["global_head_dim"],
            sliding_window=tc["sliding_window"],
            intermediate_size=tc["intermediate_size"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            num_experts=tc["num_experts"],
            top_k_experts=tc["top_k_experts"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            final_logit_softcapping=tc["final_logit_softcapping"],
            canvas_length=cfg.get("canvas_length", 256),
            layer_types=tuple(tc["layer_types"]),
            rope_parameters=tc["rope_parameters"],
        )


def rms_norm(x, weight=None, eps=1e-6):
    xf = x.float()
    normed = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        normed = normed * weight.float()
    return normed.type_as(x)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(config, layer_type, positions):
    """positions: 1D LongTensor. Returns cos, sin [len(positions), head_dim] fp32."""
    params = config.rope_parameters[layer_type]
    theta = params["rope_theta"]
    if layer_type == "sliding_attention":
        dim = config.head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    else:  # proportional partial rotary on global_head_dim
        dim = config.global_head_dim
        rotary = int(params.get("partial_rotary_factor", 1.0) * dim // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rotary, 2, dtype=torch.float32) / dim))
        inv_freq = torch.cat([inv_freq, torch.zeros(dim // 2 - rotary)])
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: [B, S, H, D]; cos/sin: [S, D]
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class DiffusionLayer:
    """One text layer, holding bf16 weights from a state dict (lazy ok)."""

    def __init__(self, config, sd, layer_idx, prefix="model.decoder.layers"):
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_kv = config.num_key_value_heads if self.is_sliding else config.num_global_key_value_heads
        p = f"{prefix}.{layer_idx}"
        # fp32 compute: CPU bf16 matmuls hit a glacial non-vectorized path.
        # Weights stay bf16-rounded (from checkpoint) so PCC vs TT is unaffected.
        g = lambda name: sd[f"{p}.{name}"].float()
        self.q_proj = g("self_attn.q_proj.weight")
        self.k_proj = g("self_attn.k_proj.weight")
        self.v_proj = g("self_attn.v_proj.weight") if self.is_sliding else None
        self.o_proj = g("self_attn.o_proj.weight")
        self.q_norm = g("self_attn.q_norm.weight")
        self.k_norm = g("self_attn.k_norm.weight")
        self.input_layernorm = g("input_layernorm.weight")
        self.post_attention_layernorm = g("post_attention_layernorm.weight")
        self.pre_feedforward_layernorm = g("pre_feedforward_layernorm.weight")
        self.post_feedforward_layernorm = g("post_feedforward_layernorm.weight")
        self.post_feedforward_layernorm_1 = g("post_feedforward_layernorm_1.weight")
        self.pre_feedforward_layernorm_2 = g("pre_feedforward_layernorm_2.weight")
        self.post_feedforward_layernorm_2 = g("post_feedforward_layernorm_2.weight")
        self.mlp_gate = g("mlp.gate_proj.weight")
        self.mlp_up = g("mlp.up_proj.weight")
        self.mlp_down = g("mlp.down_proj.weight")
        self.router_proj = g("router.proj.weight")
        self.router_scale = g("router.scale")
        self.per_expert_scale = g("router.per_expert_scale")
        self.experts_gate_up = g("experts.gate_up_proj")  # [E, 2*moe_inter, H]
        self.experts_down = g("experts.down_proj")  # [E, H, moe_inter]
        self.layer_scalar = g("layer_scalar").float().item()
        self.encoder_layer_scalar = sd[f"model.encoder.language_model.layers.{layer_idx}.layer_scalar"].float().item()

    def attention(self, x, cos, sin, attn_mask, kv_prefix=None, update_cache=None):
        """x: [B, S, H]. attn_mask: bool [S, K] or None. kv_prefix: (k, v) [B, nkv, P, D]."""
        b, s, _ = x.shape
        q = F.linear(x, self.q_proj).view(b, s, -1, self.head_dim)
        k = F.linear(x, self.k_proj).view(b, s, -1, self.head_dim)
        v = F.linear(x, self.v_proj).view(b, s, -1, self.head_dim) if self.v_proj is not None else k
        q = rms_norm(q, self.q_norm, self.config.rms_norm_eps)
        q = apply_rope(q, cos, sin).transpose(1, 2)
        v = rms_norm(v, None, self.config.rms_norm_eps).transpose(1, 2)
        k = rms_norm(k, self.k_norm, self.config.rms_norm_eps)
        k = apply_rope(k, cos, sin).transpose(1, 2)

        if update_cache is not None:
            update_cache(k, v)
        if kv_prefix is not None:
            k = torch.cat([kv_prefix[0], k], dim=2)
            v = torch.cat([kv_prefix[1], v], dim=2)

        n_rep = self.config.num_attention_heads // self.num_kv
        attn = torch.matmul(q.float(), repeat_kv(k, n_rep).float().transpose(2, 3))  # scale 1.0
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[None, None], torch.finfo(torch.float32).min)
        attn = attn.softmax(dim=-1).to(v.dtype)
        out = torch.matmul(attn, repeat_kv(v, n_rep))
        out = out.transpose(1, 2).reshape(b, s, -1)
        return F.linear(out, self.o_proj)

    def moe(self, x_flat):
        cfg = self.config
        routed_in = rms_norm(x_flat, None, cfg.rms_norm_eps) * self.router_scale * cfg.hidden_size**-0.5
        probs = F.linear(routed_in, self.router_proj).float().softmax(-1)
        top_w, top_i = torch.topk(probs, cfg.top_k_experts, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        top_w = top_w * self.per_expert_scale.float()[top_i]
        expert_in = rms_norm(x_flat, self.pre_feedforward_layernorm_2, cfg.rms_norm_eps)
        out = torch.zeros_like(expert_in)
        for e in top_i.unique():
            tok, slot = torch.where(top_i == e)
            cur = expert_in[tok]
            gate, up = F.linear(cur, self.experts_gate_up[e]).chunk(2, -1)
            h = F.gelu(gate, approximate="tanh") * up
            out.index_add_(0, tok, F.linear(h, self.experts_down[e]) * top_w[tok, slot, None].to(out.dtype))
        return out

    def forward(self, x, cos, sin, attn_mask, kv_prefix=None, update_cache=None, encoder=False):
        cfg = self.config
        residual = x
        h = rms_norm(x, self.input_layernorm, cfg.rms_norm_eps)
        h = self.attention(h, cos, sin, attn_mask, kv_prefix, update_cache)
        h = rms_norm(h, self.post_attention_layernorm, cfg.rms_norm_eps)
        x = residual + h

        residual = x
        h = rms_norm(x, self.pre_feedforward_layernorm, cfg.rms_norm_eps)
        h = F.linear(F.gelu(F.linear(h, self.mlp_gate), approximate="tanh") * F.linear(h, self.mlp_up), self.mlp_down)
        h1 = rms_norm(h, self.post_feedforward_layernorm_1, cfg.rms_norm_eps)

        flat = residual.reshape(-1, residual.shape[-1])
        h2 = self.moe(flat).reshape(residual.shape)
        h2 = rms_norm(h2, self.post_feedforward_layernorm_2, cfg.rms_norm_eps)

        h = rms_norm(h1 + h2, self.post_feedforward_layernorm, cfg.rms_norm_eps)
        x = residual + h
        return x * (self.encoder_layer_scalar if encoder else self.layer_scalar)


class DiffusionGemmaReference:
    """Text-only reference. Encoder builds per-layer dense KV; decoder denoises a canvas."""

    def __init__(self, config, sd, num_layers=None):
        self.config = config
        self.sd = sd
        self.embed = sd["model.decoder.embed_tokens.weight"].float()
        self.norm = sd["model.decoder.norm.weight"].float()
        self.sc_pre_norm = sd["model.decoder.self_conditioning.pre_norm.weight"].float()
        self.sc_gate = sd["model.decoder.self_conditioning.gate_proj.weight"].float()
        self.sc_up = sd["model.decoder.self_conditioning.up_proj.weight"].float()
        self.sc_down = sd["model.decoder.self_conditioning.down_proj.weight"].float()
        self.embed_scale = config.hidden_size**0.5
        n = num_layers or config.num_hidden_layers
        self.layers = [DiffusionLayer(config, sd, i) for i in range(n)]
        # per-layer prefix KV: list of [k, v] with shape [B, nkv, P, D]
        self.kv = [None] * n
        self.prefix_len = 0

    def _masks(self, q_pos, kv_len, causal):
        """Bool mask [S, kv_len] per layer type. q_pos: 1D positions of queries."""
        kv_pos = torch.arange(kv_len)
        full = torch.ones(len(q_pos), kv_len, dtype=torch.bool)
        if causal:
            full &= kv_pos[None, :] <= q_pos[:, None]
        sliding = full & (kv_pos[None, :] > q_pos[:, None] - self.config.sliding_window)
        return {"full_attention": full, "sliding_attention": sliding}

    def encode(self, input_ids):
        """AR-encode tokens (appended to existing prefix KV)."""
        s = input_ids.shape[-1]
        pos = torch.arange(self.prefix_len, self.prefix_len + s)
        x = F.embedding(input_ids.view(1, s), self.embed) * self.embed_scale
        x = x.to(self.embed.dtype)
        masks = self._masks(pos, self.prefix_len + s, causal=True)
        for i, layer in enumerate(self.layers):
            cos, sin = rope_cos_sin(self.config, layer.layer_type, pos)

            def upd(k, v, i=i):
                self.kv[i] = (
                    [k, v]
                    if self.kv[i] is None
                    else [torch.cat([self.kv[i][0], k], 2), torch.cat([self.kv[i][1], v], 2)]
                )

            x = layer.forward(
                x,
                cos,
                sin,
                masks[layer.layer_type][:, : self.prefix_len + s],
                kv_prefix=self.kv[i],
                update_cache=upd,
                encoder=True,
            )
        self.prefix_len += s
        return x

    def decode_canvas(self, canvas_ids, self_conditioning_logits=None, return_hidden=False):
        """One denoising forward. canvas_ids: [S]. Returns softcapped fp32 logits [S, vocab]."""
        s = canvas_ids.shape[-1]
        pos = torch.arange(self.prefix_len, self.prefix_len + s)
        x = F.embedding(canvas_ids.view(1, s), self.embed) * self.embed_scale
        x = x.to(self.embed.dtype)
        if self_conditioning_logits is not None:
            soft = self_conditioning_logits.float().softmax(-1).to(self.embed.dtype) @ self.embed * self.embed_scale
            soft = soft.view(1, s, -1)
        else:
            soft = torch.zeros_like(x)
        sc = rms_norm(soft, self.sc_pre_norm, self.config.rms_norm_eps)
        sc = F.linear(F.gelu(F.linear(sc, self.sc_gate), approximate="tanh") * F.linear(sc, self.sc_up), self.sc_down)
        x = rms_norm(x + sc, None, self.config.rms_norm_eps)

        kv_pos = torch.arange(self.prefix_len)
        for layer in self.layers:
            i = layer.layer_idx
            cos, sin = rope_cos_sin(self.config, layer.layer_type, pos)
            if layer.is_sliding:
                # HF non-compiled: canvas sees last (window - 1) prefix tokens
                lo = max(0, self.prefix_len - self.config.sliding_window + 1)
                kv_prefix = [self.kv[i][0][:, :, lo:], self.kv[i][1][:, :, lo:]] if self.kv[i] is not None else None
            else:
                kv_prefix = self.kv[i]
            x = layer.forward(x, cos, sin, None, kv_prefix=kv_prefix)

        x = rms_norm(x, self.norm, self.config.rms_norm_eps)
        if return_hidden:
            return x
        logits = F.linear(x, self.embed).float()
        cap = self.config.final_logit_softcapping
        return (torch.tanh(logits / cap) * cap).view(s, -1)
