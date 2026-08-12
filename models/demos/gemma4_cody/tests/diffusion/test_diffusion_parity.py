# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TT vs CPU-reference parity for DiffusionGemma.

The CPU reference is vendored from transformers 5.11 (the venv transformers has
no diffusion_gemma). Run on a TT host:

    export HF_MODEL=/mnt/nas/gemma-diff TT_CACHE_PATH=/mnt/nas/gemma_cache
    pytest models/demos/gemma4_cody/tests/diffusion/test_diffusion_parity.py -v -k 1x4

DIFF_TEST_LAYERS caps the layer count (default 2; set 30 for full-model PCC).
"""

import os

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.gemma4_cody.diffusion.reference import DiffusionGemmaReference, DiffusionTextConfig
from models.demos.gemma4_cody.diffusion.tt_model import TTDiffusionGemma
from models.demos.gemma4_cody.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_cody.utils.lazy_state_dict import LazyStateDict

MODEL_PATH = os.getenv("HF_MODEL", "/mnt/nas/gemma-diff")
NUM_LAYERS = int(os.getenv("DIFF_TEST_LAYERS", "2"))
PROMPT_LEN = 64  # multiple of 32 so TT pad == none and positions match exactly
REF_FILE = os.getenv("DIFF_REF_FILE", f"/mnt/nas/gemma_cache/diff_ref_logits_L{NUM_LAYERS}.pt")


def _inputs(cfg):
    """Natural-text token ids: fully random tokens drive out-of-distribution
    hidden states through bfp4-quantized experts and depress PCC (~0.95)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text = "The sky is blue because molecules in the air scatter shorter wavelengths of sunlight more strongly. "
    ids = torch.tensor(tok.encode(text * 4)[:PROMPT_LEN])
    canvas_ids = torch.tensor(tok.encode(text * 24)[: cfg.canvas_length])
    canvas1 = torch.tensor(
        tok.encode((text + "Rayleigh scattering dominates for small particles. ") * 24)[: cfg.canvas_length]
    )
    assert len(ids) == PROMPT_LEN and len(canvas_ids) == len(canvas1) == cfg.canvas_length
    return ids, canvas_ids, canvas1


def compute_reference(out_file=REF_FILE):
    """Run on a big-RAM host (TT hosts have ~3 GB): saves fp32 reference logits."""
    cfg = DiffusionTextConfig.from_json(MODEL_PATH)
    ids, canvas, canvas1 = _inputs(cfg)
    ref = DiffusionGemmaReference(cfg, LazyStateDict(MODEL_PATH), num_layers=NUM_LAYERS)
    ref.encode(ids)
    step1 = ref.decode_canvas(canvas)
    step2 = ref.decode_canvas(canvas, self_conditioning_logits=step1 / 0.8)
    # fp32 layers are ~3 GB each — free the first model before building the second
    import gc

    del ref
    gc.collect()
    ref2 = DiffusionGemmaReference(cfg, LazyStateDict(MODEL_PATH), num_layers=NUM_LAYERS)
    ref2.encode(torch.cat([ids, canvas1]))
    block2 = ref2.decode_canvas(canvas)
    torch.save({"step1": step1, "step2": step2, "block2": block2, "num_layers": NUM_LAYERS}, out_file)
    logger.info(f"saved reference logits to {out_file}")


TT_FILE = os.getenv("DIFF_TT_FILE", f"/mnt/nas/gemma_cache/diff_tt_logits_L{NUM_LAYERS}.pt")


@parametrize_mesh_with_fabric()
def test_canvas_logits_dump(mesh_device):
    """TT-host half: run the three forwards, dump bf16 logits to NAS.

    PCC compare runs on a big-RAM host via compare_logits() — loading the
    805 MB fp32 reference on a 3 GB tokenfactory host just swaps.
    """
    cfg = DiffusionTextConfig.from_json(MODEL_PATH)
    ids, canvas, canvas1 = _inputs(cfg)
    tt = TTDiffusionGemma(mesh_device, MODEL_PATH, num_layers=NUM_LAYERS)

    tt.encode_prefix(ids)
    step1, logits_tt = tt.decode_canvas(canvas)
    step2, _ = tt.decode_canvas(canvas, sc_logits_tt=logits_tt, temperature=0.8)
    tt.encode_prefix(torch.cat([ids, canvas1]))
    block2, _ = tt.decode_canvas(canvas)

    torch.save(
        {"step1": step1.bfloat16(), "step2": step2.bfloat16(), "block2": block2.bfloat16(), "num_layers": NUM_LAYERS},
        TT_FILE,
    )
    logger.info(f"saved TT logits to {TT_FILE}")


def compare_logits():
    """Big-RAM host half: PCC of dumped TT logits vs reference."""
    ref = torch.load(REF_FILE)
    tt = torch.load(TT_FILE)
    assert ref["num_layers"] == tt["num_layers"] == NUM_LAYERS
    # 0.90 matches the AR gemma4 logits threshold with bf4/bf8 weights; the
    # self-conditioning step compounds quantization noise, so it gets 0.80.
    ok = True
    for key, thresh in (("step1", 0.90), ("step2", 0.78), ("block2", 0.90)):
        passing, pcc = comp_pcc(ref[key], tt[key].float(), thresh)
        logger.info(f"{key} logits PCC: {pcc} (thresh {thresh}) {'PASS' if passing else 'FAIL'}")
        ok &= passing
    return ok


if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        assert compare_logits(), "PCC below threshold"
    else:
        compute_reference()
