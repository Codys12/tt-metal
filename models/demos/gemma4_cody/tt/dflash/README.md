# DFlash drafter (TT)

Block-diffusion speculative-decoding drafter for Gemma-4-31B. Pairs with the
same target the MTP assistant uses (hidden=5376), but drafts `block_size=8`
tokens in a single non-causal forward instead of T autoregressive passes.

## Architecture summary

| Aspect | Value |
| --- | --- |
| Layers | 5 (Qwen3-style — RMSNorm pre-norm, GQA, SwiGLU, **per-head q/k-norms at head_dim=256**, despite `model_type: llama` in config) |
| Hidden | 5376 (= target hidden, so no width-bridge projection) |
| Heads | 32 Q / 16 KV at head_dim 256 |
| MLP | intermediate 21504, SiLU/SwiGLU |
| RoPE | θ = 10000, default type |
| Block size | 8 (`block_size`) |
| Draft vocab | 32000 (narrow; remapped to target's 262144 via `d2t` offset table: `target_id = draft_id + d2t[draft_id]`) |
| Aux taps | target layers `[1, 17, 29, 47, 58]` → after `-1` offset: `[0, 16, 28, 46, 57]` |
| Mask token | id 4 |
| Verifier | `google/gemma-4-31B-it` (same as MTP assistant) |

### Checkpoint key layout (verified from `RedHatAI/gemma-4-31B-it-speculator.dflash`)

```
fc.weight                                   [5376, 26880]   bf16 — 289 MB
hidden_norm.weight                          [5376]          bf16
norm.weight                                 [5376]          bf16
embed_tokens.weight                         [262144, 5376]  bf16 — 2.8 GB (verifier embed copy)
lm_head.weight                              [32000, 5376]   bf16 — 344 MB (narrow draft head)
d2t                                         [32000]         int64 (offset: target_id = draft_id + d2t[draft_id])
t2d                                         [262144]        bool  (target id → has-draft mask)
layers.{0..4}.input_layernorm.weight        [5376]          bf16
layers.{0..4}.post_attention_layernorm.weight [5376]        bf16
layers.{0..4}.self_attn.q_proj.weight       [8192, 5376]    bf16 — 88 MB
layers.{0..4}.self_attn.k_proj.weight       [4096, 5376]    bf16 — 44 MB
layers.{0..4}.self_attn.v_proj.weight       [4096, 5376]    bf16 — 44 MB
layers.{0..4}.self_attn.o_proj.weight       [5376, 8192]    bf16 — 88 MB
layers.{0..4}.self_attn.q_norm.weight       [256]           bf16
layers.{0..4}.self_attn.k_norm.weight       [256]           bf16
layers.{0..4}.mlp.gate_proj.weight          [21504, 5376]   bf16 — 221 MB
layers.{0..4}.mlp.up_proj.weight            [21504, 5376]   bf16 — 221 MB
layers.{0..4}.mlp.down_proj.weight          [5376, 21504]   bf16 — 221 MB
```

62 keys total, 8.24 GB on disk.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | `DFlashConfig` parses HF `config.json` (incl. aux-layer `-1` offset). |
| `weights.py` | `DFlashWeights` + `load_dflash_weights()` — mesh-replicated bf16; column-parallel q/k/v + lm_head. |
| `convert_weights.py` | One-time HF safetensors → TT `.tensorbin` cache. |
| `model.py` | `DFlashDrafter` — `_layer_forward` (Llama block), top-level `forward()`. |

## Algorithm

```
target_hidden = hidden_norm(fc(concat(aux_h[0..K-1], dim=feature)))
hidden = embed_tokens([bonus, mask, mask, ..., mask])  # block_size positions
for layer in layers:
    # Pre-norm + Q/K/V projections — same k_proj/v_proj for context AND noise.
    q  = q_proj(input_layernorm(hidden))
    k  = concat([k_proj(target_hidden), k_proj(input_layernorm(hidden))], dim=seq)
    v  = concat([v_proj(target_hidden), v_proj(input_layernorm(hidden))], dim=seq)
    # RoPE: K over full (ctx + noise), Q over trailing block_size only.
    q  = rope(q, cos[-block_size:], sin[-block_size:])
    k  = rope(k, cos,                 sin)
    attn = SDPA(q, k, v, is_causal=False, scale=head_dim**-0.5)
    hidden = hidden + o_proj(attn)
    hidden = hidden + mlp(post_attention_layernorm(hidden))
draft_logits_32k = lm_head(final_norm(hidden)[:, 1:, :])  # drop bonus position
drafts_32k = argmax(draft_logits_32k, dim=-1)             # [B, block_size-1]
drafts    = drafts_32k + draft_id_to_target_id[drafts_32k]  # offset encoding, NOT direct lookup
```

The K/V cache across decode steps grows by `acceptance_length + 1` per step
(handled in Phase 5, the server integration).

## Weight conversion

```bash
cd /mnt/nas/scratch && source ./python_env/bin/activate
export TT_CACHE_PATH=/mnt/nas/gemma_cache
python -m models.demos.gemma4_cody.tt.dflash.convert_weights \
    --src /path/to/gemma-4-31B-it-speculator.dflash \
    --dst $TT_CACHE_PATH/tensor_cache_dflash_bf16
```

The HF repo for `RedHatAI/gemma-4-31B-it-speculator.dflash` ships only
`config.json` (+ a Python `config.py` for `DFlashSpeculatorConfig`) and
`model.safetensors`. Loading the reference model via
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` requires
the [`speculators`](https://github.com/vllm-project/speculators) package.

## Phase status

- **Phase 1 — base class**: ✅ `tt/drafter_base.py` landed; MTP refactored to subclass it.
- **Phase 2 — skeleton**: ✅ this module — config / weights / model / convert / reference.
- **Phase 3 — weight conversion**: ✅ converter handles the 62 actual safetensors keys; cache populated at `$TT_CACHE_PATH/tensor_cache_dflash_bf16/`.
- **Phase 4 — PCC parity vs reference**: ✅ lazy PyTorch reference runs end-to-end on real weights (peak 1.74 GB RSS via `LazyStateDict`). TT-side run is **the remaining gate** — needs a hardware run.
- **Phase 5 — server integration**: deferred until Phase 4 passes.
- **Phase 6 — byte-identity vs oracle**: deferred.
- **Phase 7 — performance**: deferred.

## Known limits (Phase 2)

- **B=1 single-user** in the forward — multi-user batched verify is a Phase 5+ concern.
- **No KV cache across decode steps** — first-decode-step shape only;
  cache management (DynamicCache-style growth + crop on rejection) lands
  with server integration.
- `propose()` raises `NotImplementedError` — call `forward()` directly for
  parity tests.
- `lm_head` is required (Gemma variant); the Qwen variant that reuses
  `target.lm_head` is not supported in this path. If we need it later,
  thread the target's lm_head through and skip the local one.
