# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side entropy-bound diffusion sampler, vendored from transformers 5.11.0
`generation_diffusion_gemma.py`. Numerics-identical, no transformers deps.

Used by both the CPU reference generate loop and the TT demo.
Defaults match /mnt/nas/gemma-diff/generation_config.json.
"""

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSamplerParams:
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    t_min: float = 0.4
    t_max: float = 0.8
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    eos_token_ids: tuple = (1, 106, 50)
    pad_token_id: int = 0


def temperature_at(params, cur_step):
    """cur_step counts down N..1."""
    return params.t_min + (params.t_max - params.t_min) * (cur_step / params.max_denoising_steps)


def token_entropy(logits, chunk=32):
    """logits: [S, V] any float dtype. Categorical entropy per token, [S].

    Row-chunked: full-canvas fp32 temporaries are ~270 MB each — too much
    for 3 GB tokenfactory hosts.
    """
    out = torch.empty(logits.shape[0])
    for i in range(0, logits.shape[0], chunk):
        lf = logits[i : i + chunk].to(torch.float32).contiguous()
        # manual stable entropy: bf16 log_softmax dispatch SIGILLs on these hosts
        lf -= lf.max(dim=-1, keepdim=True).values
        p = torch.exp(lf)
        z = p.sum(-1, keepdim=True)
        out[i : i + chunk] = (torch.log(z) - (p * lf).sum(-1, keepdim=True) / z).squeeze(-1)
    return out


class EntropyBoundSampler:
    def __init__(self, params, canvas_length, vocab_size, generator=None):
        self.p = params
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.generator = generator
        self.accepted_token_mask = None
        self._argmax_history = None

    def initialize_canvas(self):
        return torch.randint(0, self.vocab_size, (self.canvas_length,), generator=self.generator)

    def reset(self):
        self._argmax_history = None

    def step(self, current_canvas, processed_logits):
        """One acceptance+renoise step.

        processed_logits: [S, V] float (bf16 ok), already temperature-scaled.
        Row-chunked fp32 internally: full-canvas fp32 temporaries swap on the
        3 GB tokenfactory hosts.
        Returns (next_canvas, argmax_canvas, finished: bool).
        """
        s = processed_logits.shape[0]
        denoiser_canvas = torch.empty(s, dtype=torch.long)
        argmax_canvas = torch.empty(s, dtype=torch.long)
        for i in range(0, s, 32):
            probs = processed_logits[i : i + 32].float().softmax(-1)
            denoiser_canvas[i : i + 32] = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            argmax_canvas[i : i + 32] = probs.argmax(-1)

        ent = token_entropy(processed_logits)
        sorted_ent, order = torch.sort(ent)
        accept_sorted = torch.cumsum(sorted_ent, 0) - sorted_ent <= self.p.entropy_bound
        accepted = torch.zeros_like(accept_sorted).scatter(0, order, accept_sorted)
        self.accepted_token_mask = accepted

        canvas = torch.where(accepted, denoiser_canvas, current_canvas)
        canvas = torch.where(accepted, canvas, self.initialize_canvas())

        # Stable & confident stopping
        finished = False
        if self.p.stability_threshold == 0:
            stable = True
        else:
            if self._argmax_history is None:
                self._argmax_history = torch.full(
                    (self.p.stability_threshold, self.canvas_length), -1, dtype=argmax_canvas.dtype
                )
            stable = bool((self._argmax_history == argmax_canvas[None]).all())
            self._argmax_history = torch.roll(self._argmax_history, -1, 0)
            self._argmax_history[-1] = argmax_canvas
        if stable and ent.mean().item() < self.p.confidence_threshold:
            finished = True

        return canvas, argmax_canvas, finished

    def step_from_reductions(self, current_canvas, ent, argmax_canvas):
        """Acceptance+renoise step driven by device-computed reductions.

        ent: [S] temperature-scaled per-token entropy; argmax_canvas: [S] argmax
        token ids — both read back from the device (so the full [S, V] logits never
        leave the chip). Accepted (low-entropy) tokens take the argmax: they are
        near one-hot, so this matches the host multinomial in practice while
        needing no on-device RNG. Returns (next_canvas, argmax_canvas, finished).
        """
        argmax_canvas = argmax_canvas.long()
        sorted_ent, order = torch.sort(ent)
        accept_sorted = torch.cumsum(sorted_ent, 0) - sorted_ent <= self.p.entropy_bound
        accepted = torch.zeros_like(accept_sorted).scatter(0, order, accept_sorted)
        self.accepted_token_mask = accepted

        canvas = torch.where(accepted, argmax_canvas, current_canvas)
        canvas = torch.where(accepted, canvas, self.initialize_canvas())

        finished = False
        if self.p.stability_threshold == 0:
            stable = True
        else:
            if self._argmax_history is None:
                self._argmax_history = torch.full(
                    (self.p.stability_threshold, self.canvas_length), -1, dtype=argmax_canvas.dtype
                )
            stable = bool((self._argmax_history == argmax_canvas[None]).all())
            self._argmax_history = torch.roll(self._argmax_history, -1, 0)
            self._argmax_history[-1] = argmax_canvas
        if stable and ent.mean().item() < self.p.confidence_threshold:
            finished = True

        return canvas, argmax_canvas, finished


def finalize_canvas(canvas, params):
    """Replace tokens after the first EOS by pad. Returns (canvas, hit_eos)."""
    eos = torch.tensor(params.eos_token_ids)
    is_eos = torch.isin(canvas, eos)
    if not is_eos.any():
        return canvas, False
    cum = is_eos.cumsum(0)
    pad_mask = (cum > 0) & ~((cum == 1) & is_eos)
    out = canvas.clone()
    out[pad_mask] = params.pad_token_id
    return out, True
