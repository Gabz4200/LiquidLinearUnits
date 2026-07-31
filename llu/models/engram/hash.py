r"""NgramHashMapping for DeepSeek Engram O(1) multi-head N-gram hashing."""

from __future__ import annotations

import functools

import numpy as np

_prime_cache: dict[tuple[int, int], int] = {}


@functools.lru_cache(maxsize=128)
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_next_prime(start: int, seen_primes: frozenset[int]) -> int:
    candidate = start + 1
    while True:
        if _is_prime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Computes O(1) deterministic multi-head N-gram hashes for input sequences."""

    def __init__(
        self,
        engram_vocab_size: list[int] | None = None,
        max_ngram_size: int = 3,
        n_embed_per_ngram: int = 512,
        n_head_per_ngram: int = 8,
        layer_ids: list[int] | None = None,
        tokenizer_name_or_path: str | None = None,
        pad_id: int = 2,
        seed: int = 0,
        cfg=None,
    ) -> None:
        # Import here to avoid circular import
        from .tokenizer import CompressedTokenizer

        if cfg is not None:
            self.vocab_size_per_ngram = cfg.engram_vocab_size
            self.max_ngram_size = cfg.max_ngram_size
            self.n_embed_per_ngram = cfg.n_embed_per_ngram
            self.n_head_per_ngram = cfg.n_head_per_ngram
            self.layer_ids = cfg.layer_ids
            self.pad_id = cfg.pad_id
            self.seed = cfg.seed
            tokenizer_path = cfg.tokenizer_name_or_path
        else:
            self.vocab_size_per_ngram = engram_vocab_size or [129280 * 5, 129280 * 5]
            self.max_ngram_size = max_ngram_size
            self.n_embed_per_ngram = n_embed_per_ngram
            self.n_head_per_ngram = n_head_per_ngram
            self.layer_ids = layer_ids or [1, 15]
            self.pad_id = pad_id
            self.seed = seed
            tokenizer_path = tokenizer_name_or_path

        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path=tokenizer_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None and self.pad_id < len(self.compressed_tokenizer.lookup_table):
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // max(1, self.tokenizer_vocab_size))
        half_bound = max(1, m_max // 2)
        prime_1 = 10007

        self.layer_multipliers: dict[int, np.ndarray] = {}

        for layer_id in self.layer_ids:
            base_seed = int(self.seed + prime_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _calculate_vocab_size_across_layers(self) -> dict[int, list[list[int]]]:
        seen_primes: set[int] = set()
        vocab_size_across_layers: dict[int, list[list[int]]] = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes: list[list[int]] = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes: list[int] = []

                vocab_size_idx = min(ngram - 2, len(self.vocab_size_per_ngram) - 1)
                vocab_size = self.vocab_size_per_ngram[vocab_size_idx]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, frozenset(seen_primes)
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                    # Cache globally so repeated instances skip re-search
                    _prime_cache[(ngram, len(current_ngram_heads_sizes))] = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        _b, t = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x,
                ((0, 0), (k, 0)),
                mode="constant",
                constant_values=self.pad_id,
            )[:, :t]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes: list[np.ndarray] = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids: np.ndarray | list[list[int]]) -> dict[int, np.ndarray]:
        compressed_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                compressed_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers
