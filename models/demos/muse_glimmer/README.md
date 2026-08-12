# Muse Glimmer 30B

This bring-up supports the correctness-first production path only:

- batch size 1 on one TT device;
- 2,048-token chunked causal prefill into a 64-token paged KV cache;
- the checkpoint's native DFlash assistant;
- a native 16-token DFlash block in a 32-token physical target verify tile;
- greedy decoding through the OpenAI-compatible server;
- Muse's native addressed-message and ATEM tool-call protocol.

Production weights use BFP8 for attention/LM-head projections and BFP4 for
all target and DFlash MLP projections. Norms and KV caches remain BF16.
Packed verification writes only its real rows with one page-level fill per
touched page; physical padding rows never enter the target cache. Prompt-only
alignment and long-context tails may use all 32 physical rows, while native
speculative decode remains 16 tokens. A persistent
single-session prompt snapshot reuses an exact chat prefix on the next turn;
only the newly serialized assistant, tool, and user suffix is appended.

On the local single-device Blackhole server, synchronized prefill of an exactly
128,000-token prompt measured **1,549.38 prompt tok/s**; all 128,000 tokens used
the chunked paged path. A cached 64-token tool-call suffix at that position
measured 184.30 prefill tok/s and 56.01 true AR tok/s. The following 145-token
tool-response suffix (128 chunked plus a 17-token packed tail) measured 246.77
prefill tok/s and 20.13 true AR tok/s. The complete tool call and response were
parsed correctly and remained coherent. At short context, the sustained
256-token workload produced 255 true AR tokens in 1.718 seconds: **148.01 AR
tok/s**, with 240 accepted drafts across 18 packed verifies. Prefill retains
its original execution path; this speedup is isolated to captured AR replay.

The corresponding device profile measures 13.99 ms for DFlash proposal,
77.06 ms for target verification, and 1.79 ms for anchor commit. Target
matmuls account for 61.05 ms (79%) of verification after decode-only
width-sharded RMSNorm. DFlash fixed-cache writes use one contiguous packed
writer per layer, and fixed K is rotated once at the prompt boundary so each
proposal rotates only its new block. The next large optimization boundary is
therefore weight movement: smaller operation-level changes are now in the
low-single-digit-millisecond envelope per speculative cycle.

`tokens_per_second` is deliberately the end-to-end completion rate: completion
tokens divided by the entire request latency. It must not be interpreted as
prefill throughput. `ar_decode_tokens_per_second` (also exposed under the
backward-compatible `decode_tokens_per_second` name) counts only tokens produced
after the captured AR replay loop starts. It excludes tokenization, cold or
cached prefill, cache snapshots, tool-call parsing, tool-response prompt tokens,
and the first bonus token already computed by the final prefill logits. Its
denominator includes trace-input refreshes, DFlash replay, on-device greedy
sampling, packed target verification, the tiny ID readback, and anchor-commit
replay. DFlash rate varies with prompt length and draft acceptance.

## Prepare the environment

Run all commands from the `tt-metal` repository root in a Python environment
that can import `ttnn` and access one supported TT device. Install the demo's
Transformers and HTTP dependencies if they are not already present:

```bash
python -m pip install -r models/demos/muse_glimmer/requirements.txt
```

Choose paths explicitly so the procedure does not depend on the defaults in
the source tree:

```bash
export MUSE_TARGET_DIR=/path/to/Muse-Glimmer-30B
export MUSE_ASSISTANT_DIR=/path/to/Muse-Glimmer-30B-assistant
export MUSE_TT_CACHE_DIR=/path/to/writable/target-tensor-cache
export MUSE_ASSISTANT_CACHE_DIR=/path/to/writable/assistant-tensor-cache
export MUSE_MAX_SEQ_LEN=131072
export MUSE_TRACE_REGION_SIZE=256000000
```

`MUSE_TARGET_DIR` must contain the target configuration, tokenizer,
generation configuration, and safetensors checkpoint. `MUSE_ASSISTANT_DIR`
must contain the native DFlash `config.json` and safetensors checkpoint.
Both checkpoints must be fully downloaded; Hugging Face placeholder files are
not sufficient.

The target cache root is selected through `TT_CACHE_PATH`; the assistant cache
root is passed independently to the server. Neither checkpoint directory needs
to be writable:

```bash
export TT_CACHE_PATH="$MUSE_TT_CACHE_DIR"
mkdir -p "$MUSE_TT_CACHE_DIR"
mkdir -p "$MUSE_ASSISTANT_CACHE_DIR"
test -r "$MUSE_TARGET_DIR/config.json"
test -r "$MUSE_ASSISTANT_DIR/config.json"
test -w "$MUSE_TT_CACHE_DIR"
test -w "$MUSE_ASSISTANT_CACHE_DIR"
```

Use a new cache root when changing checkpoints. Target and assistant caches
must come from the exact checkpoints passed to the server.

Set `MUSE_MAX_SEQ_LEN` high enough for the tokenized prompt, requested output,
and the 16-token DFlash block. A request exceeding that capacity is rejected.

## Generate the weight caches

There is no separate conversion program. `Engine` converts and caches missing
weights while it starts. Run a one-token request to populate every production
weight cache and then exit:

```bash
python -m models.demos.muse_glimmer.server.server \
  --model-path "$MUSE_TARGET_DIR" \
  --assistant-path "$MUSE_ASSISTANT_DIR" \
  --assistant-cache-path "$MUSE_ASSISTANT_CACHE_DIR" \
  --max-seq-len "$MUSE_MAX_SEQ_LEN" \
  --trace-region-size "$MUSE_TRACE_REGION_SIZE" \
  --prompt "Reply with the word ready." \
  --max-new-tokens 1
```

The first run reads the full checkpoints and can take several minutes. Keep
enough free space for the converted weights and do not interrupt it. Later
starts reuse the tensor caches and are much faster. The command is complete
when it prints the response and DFlash timing, closes the device, and exits
successfully.

The production cache directories are:

```text
$MUSE_TT_CACHE_DIR/tensor_cache_muse_glimmer_single_device_bfp8
$MUSE_TT_CACHE_DIR/tensor_cache_muse_glimmer_single_device_bfp4
$MUSE_ASSISTANT_CACHE_DIR
```

Their contents are generated `.tensorbin` files. Re-running the command is the
safe way to finish an incomplete cache or confirm that it can be loaded.

## Start and verify the server

Use the same checkpoint, cache, and sequence-length settings used for cache
generation. Omitting `--prompt` starts the HTTP server:

```bash
export TT_CACHE_PATH="$MUSE_TT_CACHE_DIR"

python -m models.demos.muse_glimmer.server.server \
  --model-path "$MUSE_TARGET_DIR" \
  --assistant-path "$MUSE_ASSISTANT_DIR" \
  --assistant-cache-path "$MUSE_ASSISTANT_CACHE_DIR" \
  --max-seq-len "$MUSE_MAX_SEQ_LEN" \
  --trace-region-size "$MUSE_TRACE_REGION_SIZE" \
  --host 127.0.0.1 \
  --port 8000
```

Startup is ready only after weight loading completes and Uvicorn reports that
it is listening. From another shell, check health:

```bash
curl --fail --silent http://127.0.0.1:8000/health
```

Expected fields include `"status":"ok"` and
`"speculator":"native-dflash"`. Then run a generation smoke test:

```bash
curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {"role": "user", "content": "Write one short sentence about the ocean."}
    ],
    "max_tokens": 96
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

The response contains the assistant text plus `dflash.accepted_drafts`,
`dflash.elapsed_seconds`, and `dflash.tokens_per_second`. Cache reuse is visible
in `usage.prompt_tokens_details.cached_tokens`; the number actually appended is
`dflash.prefilled_prompt_tokens`. Use `dflash.prefill_tokens_per_second` for
prompt throughput and `dflash.ar_decode_tokens_per_second` for true AR
generation. `dflash.ar_decode_tokens` is the corresponding numerator. The
response also exposes `chunked_prefill_tokens`, `packed_prefill_tokens`, and the
tokenization/cache-snapshot timing so routing and overhead are observable. This
server intentionally supports one batch-1 request at a time; concurrent HTTP
requests are serialized.

Each request captures three fixed-buffer device graphs after prefill: native
DFlash proposal plus argmax, 16-token target packed verify plus argmax, and
accepted-anchor commit. Decode replays those graphs; no model, drafting,
sampling, or commit operation is dispatched eagerly in the AR interval. The
256 MB trace region above has headroom for all three graphs on one device.

## Interactive streaming chat

In a second shell, start the dependency-free interactive client against the
running server:

```bash
python -m models.demos.muse_glimmer.chat \
  --server http://127.0.0.1:8000 \
  --max-tokens 512
```

Text is printed as each accepted DFlash block arrives. The client keeps the
complete multi-turn history, automatically executes the allow-listed local
`time.get_current_time`, `calculator.calculate`, and `text.get_statistics`
tools, and submits tool results back to the model until it returns a final
answer. It prints the true AR token count/rate and cache/prefill statistics to
stderr after each model invocation. Use `/clear`, `/help`, or `/quit`; pass a
positional prompt for a streamed one-shot request. Set `MUSE_GLIMMER_URL` instead
of `--server` if preferred.

The HTTP endpoint also implements OpenAI chat-completions SSE directly. For a
raw protocol check:

```bash
curl --no-buffer --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "max_tokens": 96,
    "stream": true,
    "stream_options": {"include_usage": true}
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

The stream uses stable `chat.completion.chunk` IDs, an initial assistant-role
delta, incremental `content`/`reasoning_content`, indexed `tool_calls`, a final
`finish_reason`, an optional usage chunk, and `data: [DONE]`. The usage chunk
also carries the `dflash` performance fields.

## Tool calling and Muse tokenizer controls

Pass tools in OpenAI function-schema form. The server supplies them to the
checkpoint's native chat template and uses its declarative `response_template`
through Transformers `parse_response` to parse the generated addressed ATEM
message back into an OpenAI-compatible `tool_calls` array:

```bash
curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {
        "role": "user",
        "content": "Call weather.get_forecast for Paris for 3 days. Do not answer directly."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "weather.get_forecast",
          "description": "Get the weather forecast for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"},
              "days": {"type": "integer"}
            },
            "required": ["city", "days"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "reasoning_strength": "low",
    "max_tokens": 160
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

A tool response has `choices[0].finish_reason` set to `"tool_calls"`, null
assistant `content`, and one or more entries in
`choices[0].message.tool_calls`. Execute each call outside the server, then
submit the original messages plus the returned assistant message and a tool
message such as:

```json
{
  "role": "tool",
  "tool_call_id": "call_0",
  "content": "{\"city\":\"Paris\",\"forecast\":[\"sunny\",\"cloudy\",\"rain\"]}"
}
```

The next response normally has `finish_reason: "stop"` and final assistant
text. Tool-call history accepts `function.arguments` as either an OpenAI JSON
string or an object; the server normalizes it to the mapping required by the
Muse template. ATEM parameter values are converted with the declared JSON
schema, including integers, booleans, arrays, and objects. Multiple calls are
returned as distinct `call_N` entries. Output-length-truncated ATEM is parsed
best-effort with `finish_reason: "length"`; parser failures fall back to visible
assistant content instead of causing an HTTP 500.

Supported native template request fields are:

- `reasoning_strength`: `low`, `medium`, `high`, or `xhigh`;
- `current_date` and `knowledge_cutoff`;
- `tool_namespace_descriptions`, mapping a namespace to its description;
- `tool_choice`: `auto` or `none`.

The response exposes native `to=self` text as `reasoning_content`. A forced or
named `tool_choice` is rejected with HTTP 400 because the checkpoint template
does not provide that control. This implementation contains only the language
model: image and video message parts are rejected explicitly even though the
full Muse release has a perception encoder. Streaming is supported through
OpenAI-compatible SSE; non-greedy sampling remains outside this batch-1
correctness path. Greedy sampling is captured in both the DFlash and
target-verify traces.

The checkpoint advertises a 131,072-token context, and this bring-up has been
exercised with an exactly 128,000-token prompt, decode at that position, and a
cached multi-turn tool call plus response. TT cache capacity is fixed by
`--max-seq-len` at startup and memory use grows with it. Requests reserve the
16-token DFlash verification block and receive HTTP 400 if the tokenized prompt,
requested output, and that block exceed the configured capacity.

Prefix reuse is intentionally single-session and exact. Keep the same ordered
message history, tools, dates, knowledge cutoff, and reasoning controls from one
request to the next. A matching request reuses the prior input prompt boundary,
then chunk-prefills only the newly rendered suffix. Changing the system/template
prefix safely resets the cache.

Cached suffixes first use packed verification only until reaching the next
64-token KV-page boundary. Every complete page after that uses chunked prefill;
only the final sub-page tail returns to packed verification. Page-alignment
prefixes and long-context tails can consume 32 real prompt tokens per packed
pass, avoiding duplicate full-cache scans without changing the DFlash block.
Query tiles are selected from 256, 128, or 64 tokens according to the absolute
page position. Full-attention key tiles select the largest aligned size up to
512, and packed decode widens its per-head core split after 2,048 tokens.

For a one-shot CLI smoke test instead of HTTP, add `--prompt` and
`--max-new-tokens` to the server command. The CLI prints generated text,
DFlash acceptance, elapsed time, and tok/s before closing the device.

## Correctness and server stress tests

Correctness tests compare target prefill, packed verification, real checkpoint
layers, and native DFlash against the official Transformers implementations:

```bash
pytest -q -s models/demos/muse_glimmer/tests
```

For a quick device-free check of ATEM parsing, OpenAI response shape, invalid
input handling, and the real checkpoint chat template:

```bash
pytest -q models/demos/muse_glimmer/tests/test_server_protocol.py
```

The live server regression loads the production BFP8/BFP4 model once, performs
two native tool-call/result rounds while asserting prompt-cache reuse, sustains
a 256-token capped decode across repeated DFlash verification steps, retrieves
from a 3,221-token prompt beyond the 2,048-token sliding window, and checks
sequence-capacity rejection. It reads the same generalized path variables used
above:

```bash
export MUSE_TARGET_DIR=/path/to/Muse-Glimmer-30B
export MUSE_ASSISTANT_DIR=/path/to/Muse-Glimmer-30B-assistant

pytest -q -s models/demos/muse_glimmer/tests/test_server_e2e.py
```

The 128K test is opt-in because it allocates the full cache and performs a real
128,000-token target prefill. It then reuses that prefix for a tool call and tool
response:

```bash
export MUSE_RUN_128K=1

pytest -q -s models/demos/muse_glimmer/tests/test_long_context_128k.py
```

The chunk-boundary PCC coverage is part of the normal target-layer suite. It
compares two 128-token paged chunks covering a 256-token sequence with a single
official Torch pass for both sliding and full attention.

## Bounded performance profiling

Use the opt-in phase test instead of profiling a complete server session. It
drains model-construction markers before the selected phase, keeping the finite
device profiler buffer focused on one chunk, one packed append, or one DFlash
decode. Paths remain controlled by `MUSE_TARGET_DIR` and
`MUSE_ASSISTANT_DIR`.

Host-synchronized phase timing without Tracy:

```bash
MUSE_PROFILE_WORKLOAD=chunk \
MUSE_PROFILE_PROMPT_TOKENS=2048 \
pytest -q -s models/demos/muse_glimmer/tests/test_profile_workloads.py

MUSE_PROFILE_WORKLOAD=packed \
MUSE_PROFILE_CONTEXT_TOKENS=8192 \
MUSE_PROFILE_MAX_SEQ_LEN=12288 \
pytest -q -s models/demos/muse_glimmer/tests/test_profile_workloads.py
```

Detailed host/device Tracy capture with server phase signposts:

```bash
MUSE_TRACY_SIGNPOSTS=1 \
MUSE_PROFILE_WORKLOAD=chunk \
MUSE_PROFILE_PROMPT_TOKENS=2048 \
python tools/tracy/profile_this.py \
  --output-folder /path/to/profiler-output \
  --name-append muse_chunk \
  --command "pytest -q -s models/demos/muse_glimmer/tests/test_profile_workloads.py"

MUSE_TRACY_SIGNPOSTS=1 \
MUSE_PROFILE_WORKLOAD=decode \
MUSE_PROFILE_CONTEXT_TOKENS=64 \
MUSE_PROFILE_DECODE_TOKENS=2 \
python tools/tracy/profile_this.py \
  --output-folder /path/to/profiler-output \
  --name-append muse_decode \
  --command "pytest -q -s models/demos/muse_glimmer/tests/test_profile_workloads.py"
```

For FPU, packer, unpacker, L1, and instruction counters, use the bounded
single-layer PCC workload. `--profiler-capture-perf-counters=all` enables the
groups defined by `tt_metal/tools/profiler/perf_counters.hpp`; collection and
host/device synchronization are implemented by the TT Metal profiler.

```bash
python tools/tracy/profile_this.py \
  --output-folder /path/to/counter-output \
  --name-append muse_layer_counters \
  --profiler-capture-perf-counters=all \
  --command "pytest -q -s models/demos/muse_glimmer/tests/test_target_layer_parity.py -k two_chunk"
```

Do not enable all counter groups on the full 52-layer end-to-end server: counter
instrumentation recompiles kernels and can overflow the finite marker buffer.
Capture one bounded phase or layer, then use the ordinary Tracy phase capture
to confirm its contribution in production.
