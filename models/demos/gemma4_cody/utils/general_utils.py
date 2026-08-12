# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Gemma4 demo.
"""

from pathlib import Path

import torch
from loguru import logger


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_tensor_cache_file_path(cache_file_name, dtype, layout):
    if not cache_file_name:
        return None
    return Path(f"{cache_file_name}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def tensor_cache_exists(cache_file_name, dtype, layout):
    cache_path = get_tensor_cache_file_path(cache_file_name, dtype, layout)
    return cache_path is not None and cache_path.is_file()


def cached_tensor_placeholder(cache_file_name, dtype, layout):
    """Return a cheap placeholder when a TTNN tensor cache exists.

    ttnn.as_tensor() does not inspect the source torch tensor on cache hits, so
    this avoids materializing large HF tensors just to load the cached .tensorbin.
    """
    cache_path = get_tensor_cache_file_path(cache_file_name, dtype, layout)
    if cache_path is None or not cache_path.is_file():
        if cache_path is not None:
            logger.info(f"TTNN tensor cache not found at {cache_path}; loading HF tensor")
        return None
    logger.info(f"Using cached TTNN tensor {cache_path}; skipping HF tensor load")
    return torch.empty((), dtype=torch.bfloat16)
