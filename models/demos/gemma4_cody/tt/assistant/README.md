# Gemma 4 MTP Assistant (Drafter) on Tenstorrent

The drafter half of speculative decoding for cody. Pairs with the
**packed-decode verifier** (Approach B) measured in
`models/demos/gemma4_cody/tests/unit/test_packed_decode_compare.py`.

## Status

| Component | Status |
|---|---|
| Architecture mapping (HF → TT) | ✅ documented in `config.py`, `weights.py`, `model.py` |
| HF weights downloaded | ✅ `/mnt/nas/gemma-assistant/` (839 MB) |
| Weight conversion to TT cache layout | ✅ `convert_weights.py` ran, 48 tensors at `$TT_CACHE_PATH/tensor_cache_assistant_bf16/` |
| TT weight loader | ✅ `weights.load_drafter_weights()` — loads all 48 tensors with mesh-replicate |
| Config parser | ✅ `Gemma4AssistantConfig.from_hf_path()` |
| `Gemma4AssistantModel.__init__` | ✅ holds config + loaded weights |
| `Gemma4AssistantModel.forward` | ❌ **stubbed (`NotImplementedError`)** |
| Cross-model integration with target | ❌ not started |
| Verify + accept/reject + KV rollback | ❌ not started |

## What changes from cody's existing target-model TT code

Architecturally, the drafter is "a 4-layer Gemma 4 that consumes external
K/V." Specifically:

| Component | Target (cody current) | Drafter |
|---|---|---|
| Layer count | 60 | **4** (3 sliding + 1 full) |
| `hidden_size` | 5376 | **1024** |
| Fused QKV projection | ✅ `wqkv` | ❌ **only `q_proj`** |
| `k_proj`, `v_proj` | weight tensors | **NONE — provided by target via `shared_kv_states`** |
| `q_norm`, `k_norm` | both present | **only `q_norm`** |
| `head_dim` | 128 | **256 (sliding) / 512 (full)** |
| RoPE | per-layer-type | per-layer-type (same code reusable) |
| MLP | gate/up/down | gate/up/down (same) |
| Layer norms (4 per layer) | identical | identical |
| `layer_scalar` | per-layer | per-layer |
| `pre_projection`, `post_projection` | n/a | **NEW**: linear layers bridging drafter's 1024 ↔ target's 2816 backbone |
| `lm_head` | separate | **tied to `embed_tokens`** (no separate weight) |

## Mesh / sharding strategy

The drafter is 0.84 GB at bf16. For TP=8 on 1×8 P150, **full replication
across mesh devices is fine** (~105 MB per device). No column/row parallel
sharding needed initially — the existing `weights.py` uses
`ttnn.ReplicateTensorToMesh` throughout.

If the drafter ever becomes a bottleneck on TP=8, partial sharding can be
added (e.g., column-parallel q_proj, row-parallel o_proj like the target).
The shared_kv coming from the target is already per-device-sharded, so
the drafter consumes its device's slice directly — TP-aware sharding
shouldn't change the cross-model API.

## Implementation checklist for `forward()`

See the module docstring in `model.py` for the full algorithm. Concrete
steps in execution order:

1. `inputs_embeds = ttnn.linear(target_last_hidden, weights.pre_projection)` →
   reshape to `[B, T, hidden_size=1024]`.
2. Loop over `config.num_hidden_layers` (4):
   - `h_norm = ttnn.rms_norm(residual, weights.input_layernorm, eps)`.
   - `q = ttnn.linear(h_norm, weights.q_proj)`. Reshape to
     `[B, T, num_heads=16, head_dim_for_layer]`.
   - `q = apply_per_head_norm(q, weights.q_norm, eps)` — REUSE from
     `gemma4_cody.tt.attention.operations`.
   - `q = apply_rope(q, cos_pos, sin_pos, token_index=0)` — REUSE.
   - Get this layer's KV from `shared_kv[layer_type]` (already RoPE'd by
     the target).
   - `sdpa_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, K, V, ...)` — REUSE the same kernel cody already uses.
   - `attn_out = ttnn.linear(sdpa_out, weights.o_proj)`.
   - `attn_out = ttnn.rms_norm(attn_out, weights.post_attention_layernorm, eps)`.
   - `residual = residual + attn_out * weights.layer_scalar`.
   - `mlp_in = ttnn.rms_norm(residual, weights.pre_feedforward_layernorm, eps)`.
   - `gate = ttnn.linear(mlp_in, weights.mlp_gate)`.
   - `up = ttnn.linear(mlp_in, weights.mlp_up)`.
   - `mlp_intermediate = ttnn.silu(gate) * up`  (or whatever activation
     the config specifies — check `hidden_activation`).
   - `mlp_out = ttnn.linear(mlp_intermediate, weights.mlp_down)`.
   - `mlp_out = ttnn.rms_norm(mlp_out, weights.post_feedforward_layernorm, eps)`.
   - `residual = residual + mlp_out * weights.layer_scalar` (verify scalar
     applies here too — check HF reference).
3. `residual = ttnn.rms_norm(residual, weights.final_norm, eps)`.
4. `out_hidden = ttnn.linear(residual, weights.post_projection)` →
   `[B, T, backbone_hidden=2816]`.
5. (Optional) `logits = ttnn.linear(residual, weights.embed_tokens.T)`
   for vocab predictions (tied embeddings).

## Cross-model integration (the actual port)

The drafter doesn't run in isolation. To use it for speculation:

1. **Target's last-layer hidden state extraction.** Currently cody's
   target model finishes a decode step by computing logits and discarding
   the residual stream. To feed the drafter, we need to also export the
   per-token hidden state at the end of the final decoder layer.

2. **Target's per-layer-type "last layer of each type" KV.** Drafter
   reads from the target's KV at the deepest layer of each `layer_type`.
   For Gemma 4 26B-A4B-it the layer plan interleaves sliding+full
   somehow; we need to identify which target layer is the deepest of each
   type and read those KV slices.

3. **Drafter call site in `server.py`.** After the target's
   `_step_decode` completes, call `Gemma4AssistantModel.forward(...)` to
   get T draft tokens. Stage them for the next step's verification.

4. **Verify-and-accept logic.** Compare drafter's proposed tokens to the
   target's outputs in the next packed-decode step. Accept the
   longest-matching prefix.

5. **KV cache rollback for rejected drafts.** Tokens after the first
   rejection have already had their KV positions written by the target.
   Mark those positions invalid (overwrite on next step or zero them
   explicitly).

## File layout in this directory

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `config.py` | `Gemma4AssistantConfig` parser; matches HF config.json |
| `weights.py` | `load_drafter_weights()` + `DrafterWeights` / `DrafterLayerWeights` containers |
| `model.py` | `Gemma4AssistantModel.__init__` (wired) + `.forward` (stubbed) |
| `convert_weights.py` | Run once to convert HF safetensors → TT cache layout |
| `README.md` | This file |

## Quick sanity check

You can already verify weight loading end-to-end:

```python
from models.demos.gemma4_cody.tt.assistant.config import Gemma4AssistantConfig
from models.demos.gemma4_cody.tt.assistant.model import Gemma4AssistantModel

config = Gemma4AssistantConfig.from_hf_path("/mnt/nas/gemma-assistant")
model = Gemma4AssistantModel(mesh_device, config)
# model.weights now contains all 48 tensors as TT tensors.
# model.forward(...) raises NotImplementedError until step 2 above is done.
```

## What's a realistic next step

Implementing `forward()` is the natural next chunk. With cody's existing
reusable building blocks (apply_per_head_norm, apply_rope,
paged_scaled_dot_product_attention_decode), the forward should be ~150
lines of glue. The hard part isn't the drafter itself — it's the
cross-model integration in step 1-2 above (target hidden-state extraction
+ KV slicing). Plan on:

- 1-2 days: `forward()` standalone (with synthetic shared_kv inputs)
- 2-3 days: target-model integration (hidden state + KV plumbing)
- 1-2 days: verify+accept+rollback logic in `_step_decode`
- 1-2 days: end-to-end testing + perf tuning

Total: ~1-2 weeks of focused work for someone fluent in cody's TT layer.
