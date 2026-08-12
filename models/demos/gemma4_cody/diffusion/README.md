# DiffusionGemma on Tenstorrent

Block-diffusion text generation for **DiffusionGemma 26B-A4B** (`/mnt/nas/gemma-diff`)
running on a Blackhole P150 mesh, built as a thin wrapper over the existing
`gemma4_cody` MoE backbone.

Validated 2026-06-11 on a 4-chip P150 mesh (1×4): the full 30-layer model
answers *"Why is the sky blue?"* coherently with entropy-bounded adaptive
stopping. The denoising loop runs as a captured **ttnn device trace** with the
MoE, self-conditioning, and sampler reductions all on-device — **~0.15 s per warm
step, ~90× faster** than the original host-sampler eager path (see *Performance*).

```
=== OUTPUT (256 tokens, ~10 fwd, ~0.15 s/step) ===
The sky appears blue because of Rayleigh scattering, where sunlight interacts
with molecules and particles in the atmosphere. Shorter wavelengths, like blue
and violet, are scattered more efficiently than longer wavelengths, filling the
sky with blue light.
```

(The traced run takes a slightly different denoising trajectory than the eager
reference — both the bf16 device temperature scalars and the argmax-for-accepted
approximation perturb the entropy ranking — so it converges in 11 steps here vs
16–20 eager. All outputs are coherent; the sampler is robust to it.)

## What DiffusionGemma is

An encoder–decoder built on the Gemma-4 26B-A4B MoE stack (30 layers,
5×sliding + 1×full pattern, 128 experts / 8 active + shared MLP). Unlike an AR
LLM:

- **Encoder** autoregressively processes the prompt into a KV cache (a standard
  causal Gemma-4 forward). All text weights are **shared** with the decoder
  except a per-layer `layer_scalar` (encoder vs decoder copies differ).
- **Decoder** denoises a fixed **256-token canvas** with **bidirectional**
  self-attention, cross-attending (read-only) to the encoder KV. Sliding-window
  layers let the canvas see the last `window-1` prefix tokens; full layers see
  all of it.
- **Self-conditioning**: each denoising step feeds the previous step's logits
  back in — `softmax(logits) @ embed * scale` through a gated MLP, added to the
  canvas embeddings, then RMSNorm.
- **Sampling**: entropy-bound denoising. At each of ≤48 steps the sampler
  accepts the lowest-entropy tokens (mutual-information bound 0.1), renoises the
  rest, and stops early when predictions are stable and confident. Temperature
  decays 0.8→0.4. Defaults come from the checkpoint's `generation_config.json`.

## Files

| File | Role |
|------|------|
| `tt_model.py` | `TTDiffusionGemma` — encoder prefill (dense keep_kv), bidirectional canvas decode, on-device self-conditioning, config/state-dict adaptation over `LazyStateDict`. |
| `sampler.py` | Host-side entropy-bound sampler, linear temperature schedule, stable-and-confident stopping. Vendored from transformers 5.11. |
| `reference.py` | Pure-torch CPU reference (text-only) for PCC parity, vendored from transformers 5.11 `modeling_diffusion_gemma.py`. |
| `demo.py` | `pytest` demo: block-autoregressive generate loop. |
| `generate_weight_cache.py` | Pre-builds the TTNN weight cache (`tensor_cache_diff_bf16`) on a big-RAM host via a mock cluster. |
| `../tests/diffusion/test_diffusion_parity.py` | Split-host TT-vs-reference PCC test. |

Why vendored references: the installed `transformers` (5.9) has no
`diffusion_gemma` model type. The reference and sampler are copied from 5.11 so
parity tests need no transformers upgrade. `config.json` is parsed directly
(AutoConfig can't load an unknown `model_type`).

## Running

Environment (Tenstorrent host with `/mnt/nas` mounted):

```bash
cd /mnt/nas/scratch2 && source ./python_env/bin/activate
export TT_METAL_HOST_CHANNEL_SIZE_MB=64 TT_METAL_HOME=/mnt/nas/scratch2
export HF_MODEL=/mnt/nas/gemma-diff TT_CACHE_PATH=/mnt/nas/gemma_cache
export MESH_DEVICE=8xP150 TT_VISIBLE_DEVICES=0,2,3,5
```

**Weight cache** (one-time; on a big-RAM host — tokenfactory hosts have ~3 GB).
The first build materializes the 1.5 GB embed/lm_head torch tensors, which a TT
host can't hold, so run it on genswarm1 against a mock cluster matching the
inference TP:

```bash
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/mnt/nas/1x4.yaml
python models/demos/gemma4_cody/diffusion/generate_weight_cache.py --mesh-shape 1x4
```

**Demo**:

```bash
pytest models/demos/gemma4_cody/diffusion/demo.py -k 1x4 -q -s --timeout=9900
```

Overrides: `DIFF_NUM_LAYERS` (cap layers), `DIFF_MAX_NEW_TOKENS` (default 256 =
1 canvas), `DIFF_MAX_STEPS` (default 48), `DIFF_PROMPT`, `DIFF_SEED`.

## Parity (two-host split)

The fp32 reference (805 MB of logits) swaps a 3 GB TT host, so parity is split:
dump TT logits on the TT host, compare on a big-RAM host.

```bash
# 1. Reference logits — big-RAM host (genswarm1), mock cluster for the import
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/mnt/nas/1x4.yaml DIFF_TEST_LAYERS=2
python -c "from models.demos.gemma4_cody.tests.diffusion.test_diffusion_parity import compute_reference; compute_reference()"

# 2. TT logits dump — TT host
DIFF_TEST_LAYERS=2 pytest models/demos/gemma4_cody/tests/diffusion/test_diffusion_parity.py -k 1x4 -q -s

# 3. Compare — big-RAM host
DIFF_TEST_LAYERS=2 python models/demos/gemma4_cody/tests/diffusion/test_diffusion_parity.py --compare
```

PCC bands (bf4 expert / bf8 attention weights, same as the AR `gemma4_cody`
repo's own ≥0.90 logits threshold): step-1 ~0.92–0.94, self-conditioning step
~0.80 (compounds quantization noise), block-2 ~0.94. Use natural-text token ids,
not uniform-random — random tokens drive out-of-distribution hidden states
through the quantized experts and depress PCC.

## Implementation notes / gotchas

- **M=256 matmuls**: the canvas is 256 rows; `SharedMLP`'s L1 block configs and
  the lm_head's 2D-mcast factory both assume the ≤128 prefill / 32 decode
  shapes. The FFN runs in 128-row halves and the lm_head in 32-row chunks.
- **`to_layout` on the 262k-row embedding stalls**: the self-conditioning
  soft-embedding matmul needs a TILE-layout copy of the tied embedding. Built
  once as a cached `as_tensor` (`embed_tokens_tile_*`), not a runtime
  `to_layout`.
- **RMSNorm gamma** must be `[1,1,H/32,32]` row-major, not `[1,1,1,H]`.
- **bf16 `log_softmax` SIGILLs** on the tokenfactory CPUs — the sampler computes
  entropy via a manual fp32 stable logsumexp, row-chunked to bound memory.
- **No paged KV**: the prefix is re-encoded each block (fine for the 256-canvas
  demo). Follow-up: chunked-prefill the prefix.

## Device trace

The denoising inner loop runs the same 30-layer forward every step with only the
canvas token ids and the self-conditioning temperature changing, so it is
captured once as a ttnn trace and replayed (`tt_model.py`:
`decode_prepare`/`decode_step`/`decode_capture`/`decode_replay`/`decode_finish`).

- **Step 1** (zeros self-conditioning) and **step 2** (real SC) run eagerly:
  step 1 has a different op graph, step 2 is the compile run that warms the
  traced kernels. The trace is captured lazily before step 3 and every later step
  is one `execute_trace`.
- **Per-step dynamic inputs** without breaking the captured graph: canvas ids →
  persistent uint32 buffer (`copy_host_to_device` each step); two temperatures
  (self-conditioning, sampler) → persistent `[1,1,1,1]` device scalars
  broadcast-multiplied where used (so the op graph is constant while the values
  vary); previous logits → a persistent `sc_buf` read at the top of SC. tt-metal
  **forbids data writes during capture** (`TT_FATAL: Writes are not supported
  during trace capture` — also fires on `from_torch(device=)` inside capture), so
  the SC feedback (`logits → sc_buf`) is an *eager* device copy done after each
  replay, not inside the trace.
- **On-device reduction.** The lm_head logits are `[256, 262144]` bf16 (134 MB).
  Reading them to the host and running the entropy-bound sampler there is the
  dominant cost. Instead the trace computes the per-token **argmax** (token id,
  temperature-independent) and the temperature-scaled **entropy** on-device
  (`ttnn.argmax` + an fp32-stable `log(Z) − Σp·lf` over the vocab), and only the
  two `[256]` reductions are read back. The host then does the cheap part — sort
  the 256 entropies, the cumulative-entropy accept mask, renoise, and the
  stability/confidence stop (`sampler.step_from_reductions`). Accepted tokens take
  the argmax rather than a multinomial draw: accepted tokens are near one-hot
  (their cumulative entropy is ≤ 0.1 across ~200 of them), so this matches the
  multinomial in practice and needs no on-device RNG. The full 134 MB logits never
  leave the chip.
- One trace per canvas (the prefix KV/rope/masks are constant within a canvas);
  released and re-captured per canvas as the prefix grows.

## Performance — warm trace step, 13.5 s → 0.15 s (~90×)

Measured (4-chip 1×4, 256-canvas demo), each step a single `execute_trace`:

| stage | warm s/step | what changed |
|------|-------------|--------------|
| eager (host sampler) | ~13.5 | baseline |
| trace + on-device reduction | ~8.4 | kill the 134 MB logits→host round-trip + host sampler |
| + `in0_block_w=8` | ~3.9 | MoE gate/up were K-starved (88 serial K-steps) |
| + concat-experts MoE | ~2.4 | gate/up as one big-N matmul → full core grid |
| + reduce-sharding | ~0.91 | argmax/entropy on the sharded `[S, V/tp]` logits |
| + argmax topk, M=256 MoE, concat-down | ~0.34 | `ttnn.topk(k=1)` not 65536-wide argmax (505→19 ms); MoE once/layer; fold routing → one down matmul |
| + expand-matmul routing fold | ~0.22 | `routing @ [E,E·I]` block matrix instead of two wide-`g` tile-repack reshapes |
| + block-sharded down matmul | ~0.20 | the K=24576 down's auto config serialized K (1.36→0.53 ms) |
| + lm_head as one M=256 matmul | ~0.19 | the 8×32-row chunking was an unneeded stale assumption |
| + V-sharded self-conditioning | ~0.16 | drop the 134 MB logits all-gather; SC embmm 11.8→3.5 ms (K 262144→65536) |
| **+ shared MLP at M=256** | **~0.15** | one pass + one all-reduce, not 2× M=128 halves (2.7×) |

The diagnosis mattered more than any single fix. **Profiling a *compile* step lies**
— the first real-SC forward JIT-compiles every kernel, so a naive per-op timer
shows self-conditioning at 74% when warm it's 28 ms. Profiling a *warm* step (env
`DIFF_NO_TRACE` + `DIFF_PROFILE`, on a post-compile step) is what exposed the real
bottlenecks: MoE 73%, reduce 21%.

- **Why the trace alone (13.5→8.4) is modest:** a trace removes host op-*dispatch*,
  but at M=256 ttnn's async dispatch already overlaps host enqueue behind device
  execution. The real win there was killing the 134 MB device→host logits read and
  the host entropy/multinomial/sort, by computing argmax + entropy on-device
  (`total == fwd`).
- **MoE (the big one).** The gemma4 expert path computes ALL 128 experts (gpt_oss
  dense pattern) as 128 *serial per-expert* `sparse_matmul`s at M=32 — ~99% of its
  6.7 s was utilization loss, not FLOP. Two fixes: (1) `in0_block_w=8` so the
  K=2816 contraction isn't 88 serial single-tile steps; (2) **concat-experts** —
  relayout gate/up to `[H, E·I]` (`_build_moe_concat`, a one-time bf4→bf16→permute
  →bf4 requant since permute rejects bf4) so all experts are ONE matmul with
  N=24576 = 704 tiles → the full grid. `f_moe` 6746 → **529 ms**.
- **reduce-sharding.** argmax + entropy ran on the full all-gathered `[256, 262144]`
  logits, 4× redundantly across TP. Instead compute per-shard partials
  (`a_idx, a_val, Z, SX`) on the *pre-all-gather* `[256, V/tp]` logits and combine
  cross-shard on the host (`_reduce_partials` / `_combine_partials`) — exact, 4×
  less width. ~1.96 s → the step lands at **0.91 s**.

Numerics: the argmax-for-accepted, bf16-entropy, and reduce-sharding paths all
match the reference (`DIFF_REDUCE_CHECK=1`: argmax 256/256, entropy meanerr 0.005;
the 0.07 maxerr is on high-entropy *rejected* tokens, off the accept boundary). The
traced trajectory differs slightly from the eager reference (9–11 forwards vs
16–20; all coherent). Knobs (all default to the fast path): `DIFF_MOE_CONCAT`,
`DIFF_REDUCE_SHARD`, `DIFF_MOE_IN0BW`, `DIFF_ENTROPY_FP32`, `DIFF_REDUCE_CHECK`,
`DIFF_PROFILE`/`DIFF_NO_TRACE`/`DIFF_MOE_PROF` (warm per-op profiling).

**0.91 → 0.15 s (sprint 3, all exact — logits bit-identical, `DIFF_REDUCE_CHECK`
256/256, coherent).** The critical lesson: the `DIFF_NO_TRACE` *serial* profile sum
(~570 ms) badly over-predicts the *trace* step (~150 ms) — trace overlaps host
dispatch, so an op's serial time is meaningless until you measure its **trace**
contribution by zeroing it (`DIFF_SKIP_*`) or via `DIFF_NUM_LAYERS=1`. That A/B
testing showed attn (143 ms serial → ~15 ms trace), shared MLP (269 → ~20), and the
router (105 → ~5–10) are almost fully overlap-hidden, while the *unoverlapped* wins
were: the wide-`g` routing-fold reshapes (replaced by an expand matmul), the down
matmul's serialized K=24576 (block-sharded config), the full-vocab SC and its
134 MB all-gather (V-sharded SC + `embed_shard`), the needless lm_head chunking, and
the shared-MLP halving (run at M=256). Decomposed, the ~0.15 s step is ~103 ms
per-layer (30 × 3.45 ms, matmul-bound and measured near-optimal — a token-gather
*sparse* MoE was benchmarked 2–12× **slower**, `bench_moe_matmul.py`) + ~47 ms tail
(the topk reduce is ~19 ms; a topk-free max+arg trick is ~12 ms cheaper but not
bit-exact, so the exact topk is kept — sampling must not drift). Below ~0.15 s needs
model-level change (layer count, the 262144 vocab, kernel fusion), not op-tuning.

Remaining levers: the prefix is re-encoded per block (chunked-prefill would help
multi-canvas). Sprint bench probes: `bench_moe_matmul.py`, `bench_down_config.py`,
`bench_tail_configs.py`, `bench_shared_config.py`.

See also `~/.claude/.../memory/tokenfactory-nas-topology.md` for the NAS/host
layout (genswarm1 NFS server, IPv4-only exports).
