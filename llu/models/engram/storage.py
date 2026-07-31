r"""Offloaded EngramEmbeddingStore supporting CPU and Disk (SSD mmap) backends."""

from __future__ import annotations

import os
import tempfile
from typing import Literal, cast

import numpy as np
import torch
from torch import nn


class EngramEmbeddingStore(nn.Module):
    """Embedding storage manager for DeepSeek Engram.

    Supports zero-copy CPU offloading when compute runs on GPU, and fast SSD
    memory-mapped disk storage when compute runs on CPU.
    """

    def __init__(
        self,
        list_of_N: list[int],
        D: int,
        storage_backend: Literal["auto", "cpu", "disk", "cuda"] = "auto",
        disk_dir: str | None = None,
        target_device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.list_of_N = list_of_N
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.target_dtype = dtype
        self.target_device = target_device or torch.device("cpu")

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long, device=self.target_device)
        )
        self.total_N = sum(list_of_N)

        # Resolve backend
        if storage_backend == "auto":
            if self.target_device.type == "cuda":
                resolved_backend = "cpu"
            else:
                resolved_backend = "disk"
        else:
            resolved_backend = storage_backend

        self.backend = resolved_backend
        self.disk_dir = disk_dir
        self.disk_filepath: str | None = None
        self.memmap: np.memmap | None = None
        self.cpu_embedding: nn.Embedding | None = None
        self.device_embedding: nn.Embedding | None = None

        self._init_storage()

    def _init_storage(self) -> None:
        std = 1.0 / max(1, self.embedding_dim) ** 0.5

        if self.backend == "disk":
            if self.disk_dir is None:
                self.disk_dir = tempfile.gettempdir()
            os.makedirs(self.disk_dir, exist_ok=True)
            self.disk_filepath = os.path.join(
                self.disk_dir, f"engram_embed_{id(self)}_{self.total_N}x{self.embedding_dim}.bin"
            )
            # Create / open memory mapped binary file using chunked writes
            shape = (self.total_N, self.embedding_dim)
            if not os.path.exists(self.disk_filepath):
                fp = np.memmap(self.disk_filepath, dtype="float32", mode="w+", shape=shape)
                chunk_size = 65536
                for i in range(0, self.total_N, chunk_size):
                    n = min(chunk_size, self.total_N - i)
                    fp[i : i + n] = np.random.normal(0.0, std, size=(n, self.embedding_dim)).astype(
                        np.float32
                    )
                fp.flush()
                del fp

            self.memmap = np.memmap(self.disk_filepath, dtype="float32", mode="r", shape=shape)

        elif self.backend == "cpu":
            self.cpu_embedding = nn.Embedding(
                num_embeddings=self.total_N,
                embedding_dim=self.embedding_dim,
                device=torch.device("cpu"),
            )
            nn.init.normal_(self.cpu_embedding.weight, mean=0.0, std=std)

        else:
            # "cuda" or standard target_device embedding
            self.device_embedding = nn.Embedding(
                num_embeddings=self.total_N,
                embedding_dim=self.embedding_dim,
                device=self.target_device,
            )
            nn.init.normal_(self.device_embedding.weight, mean=0.0, std=std)

    def is_on_disk(self) -> bool:
        return self.backend == "disk" and self.memmap is not None

    def is_on_cpu_table(self) -> bool:
        return self.backend == "cpu" and self.cpu_embedding is not None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        offsets = cast(torch.Tensor, self.offsets)
        shifted_input_ids = input_ids.to(self.target_device) + offsets

        if self.backend == "disk" and self.memmap is not None:
            # SSD mmap lookup
            ids_np = shifted_input_ids.detach().cpu().numpy()
            np.clip(ids_np, 0, self.total_N - 1, out=ids_np)
            fetched_np = np.asarray(self.memmap[ids_np], dtype=np.float32)
            out = torch.from_numpy(fetched_np).to(
                device=self.target_device, dtype=self.target_dtype
            )
            return out

        if self.backend == "cpu" and self.cpu_embedding is not None:
            # CPU table lookup
            ids_cpu = shifted_input_ids.detach().cpu()
            ids_cpu.clamp_(0, self.total_N - 1)
            with torch.no_grad():
                cpu_out = self.cpu_embedding(ids_cpu)
            return cpu_out.to(device=self.target_device, dtype=self.target_dtype)

        if self.device_embedding is not None:
            ids_dev = shifted_input_ids.clamp(0, self.total_N - 1)
            return self.device_embedding(ids_dev).to(dtype=self.target_dtype)

        raise RuntimeError(f"EngramEmbeddingStore has uninitialized backend {self.backend}")

    def close(self) -> None:
        if getattr(self, "memmap", None) is not None:
            try:
                if hasattr(self.memmap, "_mmap") and self.memmap._mmap is not None:
                    self.memmap._mmap.close()
            except (AttributeError, OSError):
                pass
            object.__setattr__(self, "memmap", None)
        if self.disk_filepath and os.path.exists(self.disk_filepath):
            try:
                os.remove(self.disk_filepath)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()
