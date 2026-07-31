r"""Configuration classes for DeepSeek Engram memory module."""

from dataclasses import dataclass, field


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str | None = None
    engram_vocab_size: list[int] = field(default_factory=lambda: [10000, 10000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: list[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    storage_backend: str = "auto"  # "auto" | "cpu" | "disk" | "cuda"
    disk_dir: str | None = None
    engram_mode: str = "cond"  # "cond" | "additive" | "both"


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
