r"""Tests for DeepSeek Engram implementation in ``llu.models.engram``."""

import os
import tempfile

import numpy as np
import torch

from llu.models.engram import (
    CompressedTokenizer,
    Engram,
    EngramConfig,
    EngramEmbeddingStore,
    NgramHashMapping,
)
from llu.models.liquid_llm import build_llm
from llu.models.liquid_model import build_model
from llu.models.llns.stable_liquid import StableLiquidLN


def test_when_compressed_tokenizer_called_then_compresses_input_ids():
    input_ids = np.array([[10, 20, 30], [40, 50, -1]], dtype=np.int64)
    # Using fallback/mock lookup table when tokenizer is None
    comp_tok = CompressedTokenizer(tokenizer_name_or_path=None, vocab_size=100)
    out = comp_tok(input_ids)
    assert out.shape == input_ids.shape
    assert out[1, 2] == -1  # padding / mask preserved


def test_when_ngram_hash_mapping_called_then_produces_layer_hashes():
    layer_ids = [1, 5]
    hash_mapping = NgramHashMapping(
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_head_per_ngram=2,
        layer_ids=layer_ids,
        tokenizer_name_or_path=None,
        pad_id=0,
        seed=42,
    )
    input_ids = np.array([[1, 5, 10, 15], [2, 6, 12, 20]], dtype=np.int64)
    hashes = hash_mapping.hash(input_ids)
    assert 1 in hashes and 5 in hashes
    # max_ngram=3 means n=2 (2 heads) and n=3 (2 heads) => 4 heads total
    assert hashes[1].shape == (2, 4, 4)


def test_when_offloaded_embedding_on_gpu_mock_then_table_on_cpu():
    # Test CPU offload mode (simulating GPU execution device)
    list_of_N = [100, 200]
    D = 16
    store = EngramEmbeddingStore(
        list_of_N=list_of_N,
        D=D,
        storage_backend="cpu",
        target_device=torch.device("cpu"),
    )
    indices = torch.tensor([[[0, 1], [5, 6]], [[10, 11], [150, 151]]], dtype=torch.long)
    out = store(indices)
    assert out.shape == (2, 2, 2, D)
    assert store.is_on_cpu_table()


def test_when_offloaded_embedding_on_cpu_then_table_on_disk():
    # Test Disk offload mode for CPU computation
    with tempfile.TemporaryDirectory() as tmpdir:
        list_of_N = [50, 50]
        D = 8
        store = EngramEmbeddingStore(
            list_of_N=list_of_N,
            D=D,
            storage_backend="disk",
            disk_dir=tmpdir,
            target_device=torch.device("cpu"),
        )
        indices = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.long)
        out = store(indices)
        assert out.shape == (2, 2, 2, D)
        assert store.is_on_disk()
        assert store.disk_filepath is not None and os.path.exists(store.disk_filepath)


def test_when_engram_forward_called_then_returns_out_and_cond():
    cfg = EngramConfig(
        layer_ids=[1],
        tokenizer_name_or_path=None,
        max_ngram_size=3,
        n_embed_per_ngram=32,
        n_head_per_ngram=2,
    )
    engram_module = Engram(
        layer_id=1,
        cfg=cfg,
        hidden_size=64,
        hc_mult=2,
        storage_backend="cpu",
    )
    B, T = 2, 8
    hidden_states = torch.randn(B, T, 2, 64)
    input_ids = torch.randint(0, 100, (B, T))

    output, cond = engram_module(hidden_states, input_ids)
    assert output.shape == (B, T, 2, 64)
    assert cond.shape == (B, T, 64)
    assert torch.isfinite(output).all()
    assert torch.isfinite(cond).all()


def test_when_stable_liquid_conditioned_on_engram_retrieval_then_forward_ok():
    s_liquid = StableLiquidLN(in_features=64, out_features=64, cond_dim=64)
    x = torch.randn(2, 8, 64)
    engram_cond = torch.randn(2, 8, 64)
    out = s_liquid(x, cond=engram_cond)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_when_liquid_transformer_with_engram_then_forward_ok():
    for mode in ["cond", "additive", "both"]:
        model = build_model(
            "StableLiquidLN",
            d_model=32,
            out_dim=10,
            num_layers=2,
            use_engram=True,
            engram_mode=mode,
            engram_layer_ids=[0],
            tokenizer_name_or_path=None,
        )
        x = torch.randn(2, 8, 32)
        input_ids = torch.randint(0, 100, (2, 8))
        out = model(x, input_ids=input_ids)
        assert out.shape == (2, 8, 10)
        assert torch.isfinite(out).all()


def test_when_liquid_llm_with_engram_then_forward_ok():
    for mode in ["cond", "additive", "both"]:
        model = build_llm(
            "ours",
            "tiny",
            lln="StableLiquidLN",
            use_engram=True,
            engram_mode=mode,
            engram_layer_ids=[0],
            tokenizer_name_or_path=None,
        )
        idx = torch.randint(0, 50257, (2, 16))
        out = model(idx)
        assert out.shape == (2, 16, 50257)
        assert torch.isfinite(out).all()
