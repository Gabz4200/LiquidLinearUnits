r"""CompressedTokenizer for DeepSeek Engram N-gram key normalization."""

from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=16)
def _get_cached_lookup_table(
    tokenizer_name_or_path: str | None, default_vocab_size: int
) -> tuple[np.ndarray, int]:
    if tokenizer_name_or_path is not None:
        try:
            from tokenizers import Regex, normalizers
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            if tokenizer is not None:
                sentinel = "\ue000"
                normalizer = normalizers.Sequence(
                    [
                        normalizers.NFKC(),
                        normalizers.NFD(),
                        normalizers.StripAccents(),
                        normalizers.Lowercase(),
                        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                        normalizers.Replace(Regex(r"^ $"), sentinel),
                        normalizers.Strip(),
                        normalizers.Replace(sentinel, " "),
                    ]
                )
                old2new: dict[int, int] = {}
                key2new: dict[str, int] = {}
                new_tokens: list[str] = []

                tok_vocab_size = len(tokenizer)
                for tid in range(tok_vocab_size):
                    text = tokenizer.decode([tid], skip_special_tokens=False)
                    if "\ufffd" in text:
                        raw_key = tokenizer.convert_ids_to_tokens(tid)
                        key: str = raw_key[0] if isinstance(raw_key, list) else str(raw_key)
                    else:
                        norm = normalizer.normalize_str(text)
                        key = norm if norm else text

                    nid = key2new.get(key)
                    if nid is None:
                        nid = len(new_tokens)
                        key2new[key] = nid
                        new_tokens.append(key)
                    old2new[tid] = nid

                lookup = np.empty(tok_vocab_size, dtype=np.int64)
                for tid in range(tok_vocab_size):
                    lookup[tid] = old2new[tid]

                return lookup, len(new_tokens)
        except (ImportError, OSError, ValueError, KeyError, AttributeError):
            pass

    lookup = np.arange(default_vocab_size, dtype=np.int64)
    return lookup, default_vocab_size


class CompressedTokenizer:
    """Compresses token IDs into normalized token IDs for O(1) N-gram hashing.

    Maps original vocabulary IDs into normalized token equivalence classes so
    that morphological variations (case, whitespace, accents) hash to identical
    N-gram keys.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str | None = None,
        vocab_size: int = 129280,
    ) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.default_vocab_size = vocab_size
        self.lookup_table, self.num_new_token = _get_cached_lookup_table(
            tokenizer_name_or_path, vocab_size
        )

    def __len__(self) -> int:
        return self.num_new_token

    def compress(self, input_ids: np.ndarray | list[list[int]]) -> np.ndarray:
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        valid_ids = np.clip(valid_ids, 0, len(self.lookup_table) - 1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids: np.ndarray | list[list[int]]) -> np.ndarray:
        return self.compress(input_ids)
