# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy, pread-based view over HuggingFace safetensors shards.

Tensors are materialized on __getitem__ only, by reading just the bytes for
the requested tensor out of the shard file. This avoids the whole-file mmap
that safetensors.safe_open does — critical on hosts where a 47 GiB VMA
reservation is rejected by the kernel (ENOMEM from mmap).

fp32 tensors are cast to bf16 at access time (matching the eager loader).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# safetensors dtype string -> (torch dtype, bytes per element)
_DTYPE_MAP: dict[str, tuple[torch.dtype, int]] = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I32": (torch.int32, 4),
    "F32": (torch.float32, 4),
    "I64": (torch.int64, 8),
    "F64": (torch.float64, 8),
}


class _ShardReader:
    """Header parse + seek/readinto-backed tensor fetch for a single safetensors file."""

    def __init__(self, path: Path):
        self._path = path
        self._file = open(path, "rb")
        header_len_bytes = self._file.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Truncated safetensors header in {path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_json = self._file.read(header_len).decode("utf-8")
        self._header = json.loads(header_json)
        self._header.pop("__metadata__", None)
        self._data_start = 8 + header_len

    def keys(self) -> list[str]:
        return list(self._header.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        meta = self._header[key]
        dtype_str = meta["dtype"]
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for key {key!r}")
        torch_dtype, _elem_size = _DTYPE_MAP[dtype_str]
        begin, end = meta["data_offsets"]
        nbytes = end - begin
        shape = tuple(meta["shape"])

        if nbytes == 0:
            return torch.empty(shape, dtype=torch_dtype)

        # readinto writes directly into the bytearray — no intermediate bytes copy.
        buf = bytearray(nbytes)
        view = memoryview(buf)
        self._file.seek(self._data_start + begin)
        read = 0
        while read < nbytes:
            n = self._file.readinto(view[read:])
            if not n:
                raise IOError(f"Short read for {key!r} in {self._path} at byte {read} of {nbytes}")
            read += n

        tensor = torch.frombuffer(buf, dtype=torch_dtype)
        return tensor.view(*shape) if shape else tensor.reshape(())

    def get_tensor_rows(self, key: str, row_indices: torch.Tensor) -> torch.Tensor:
        """Read a sparse selection of rows from a 2-D tensor without materializing the full matrix."""
        meta = self._header[key]
        dtype_str = meta["dtype"]
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for key {key!r}")
        torch_dtype, elem_size = _DTYPE_MAP[dtype_str]
        shape = tuple(meta["shape"])
        if len(shape) != 2:
            raise ValueError(f"get_tensor_rows requires a 2-D tensor; {key!r} has shape {shape}")
        rows, cols = shape
        begin, _end = meta["data_offsets"]
        row_nbytes = cols * elem_size
        n_select = int(row_indices.numel())
        out = bytearray(n_select * row_nbytes)
        out_view = memoryview(out)
        for i, idx in enumerate(row_indices.tolist()):
            if idx < 0 or idx >= rows:
                raise IndexError(f"row index {idx} out of range for {key!r} with {rows} rows")
            self._file.seek(self._data_start + begin + idx * row_nbytes)
            slot = out_view[i * row_nbytes : (i + 1) * row_nbytes]
            read = 0
            while read < row_nbytes:
                n = self._file.readinto(slot[read:])
                if not n:
                    raise IOError(f"Short read for row {idx} of {key!r} in {self._path}")
                read += n
        return torch.frombuffer(out, dtype=torch_dtype).view(n_select, cols)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class LazyStateDict(Mapping[str, torch.Tensor]):
    """
    Read-only Mapping[str, torch.Tensor] that loads tensors from safetensors
    shards lazily via pread. One file descriptor and one header dict per shard
    is kept; tensor data is read on demand and not cached, so peak RSS tracks
    what the caller holds rather than the full checkpoint.
    """

    def __init__(
        self,
        model_path: Path,
        base_prefix: str = "",
        *,
        _full_to_file: Optional[dict[str, str]] = None,
        _readers: Optional[dict[str, _ShardReader]] = None,
    ):
        self._model_path = Path(model_path)
        self._base_prefix = base_prefix
        self._readers: dict[str, _ShardReader] = {} if _readers is None else _readers

        if _full_to_file is None:
            index_path = self._model_path / "model.safetensors.index.json"
            if index_path.is_file():
                with index_path.open("r", encoding="utf-8") as f:
                    self._full_to_file = dict(json.load(f)["weight_map"])
            else:
                single = self._model_path / "model.safetensors"
                if not single.is_file():
                    raise FileNotFoundError(
                        f"No model.safetensors.index.json or model.safetensors in {self._model_path}"
                    )
                reader = self._get_reader(single.name)
                self._full_to_file = {k: single.name for k in reader.keys()}
            num_keys = len(self._full_to_file)
            num_files = len(set(self._full_to_file.values()))
            logger.info(f"LazyStateDict initialized: {num_keys} keys across {num_files} shards")
        else:
            self._full_to_file = _full_to_file

    def _get_reader(self, filename: str) -> _ShardReader:
        reader = self._readers.get(filename)
        if reader is not None:
            return reader
        reader = _ShardReader(self._model_path / filename)
        self._readers[filename] = reader
        return reader

    def close(self) -> None:
        for filename, reader in list(self._readers.items()):
            try:
                reader.close()
            except Exception as e:
                logger.error(f"Failed to close reader for '{filename}': {e}")
        self._readers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _full_key(self, key: str) -> str:
        return self._base_prefix + key

    def substate(self, key: str) -> "LazyStateDict":
        return LazyStateDict(
            self._model_path,
            base_prefix=f"{self._full_key(key)}.",
            _full_to_file=self._full_to_file,
            _readers=self._readers,
        )

    def __getitem__(self, key: str) -> torch.Tensor:
        full_key = self._full_key(key)
        filename = self._full_to_file.get(full_key)
        if filename is None:
            raise KeyError(key)
        logger.info(f"Loading HF tensor {full_key} from {filename}")
        tensor = self._get_reader(filename).get_tensor(full_key)
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)
        logger.info(f"Loaded HF tensor {full_key} with shape {tuple(tensor.shape)} and dtype {tensor.dtype}")
        return tensor

    def get_tensor_rows(self, key: str, row_indices: torch.Tensor) -> torch.Tensor:
        """
        Read only the requested rows of a 2-D tensor (e.g. embed_tokens.weight) instead
        of materializing the full matrix. Mirrors the fp32->bf16 cast of __getitem__.
        """
        full_key = self._full_key(key)
        filename = self._full_to_file.get(full_key)
        if filename is None:
            raise KeyError(key)
        tensor = self._get_reader(filename).get_tensor_rows(full_key, row_indices)
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.bfloat16)
        return tensor

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._full_key(key) in self._full_to_file

    def __iter__(self) -> Iterator[str]:
        base = self._base_prefix
        for full_key in self._full_to_file:
            if full_key.startswith(base):
                yield full_key[len(base) :]

    def __len__(self) -> int:
        if not self._base_prefix:
            return len(self._full_to_file)
        return sum(1 for _ in self)
