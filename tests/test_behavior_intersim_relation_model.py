# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the InterSim relation-only VectorNet."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="Relation model tests require the tactics2d[behavior] extra.")

import torch

from tactics2d.behavior.intersim.relation_model import (
    _MLP,
    RelationVectorNet,
    _CrossAttention,
    _DecoderResCat,
    _GlobalGraph,
    _GlobalGraphRes,
    _LayerNorm,
    _merge_sub_graph,
    _merge_tensors,
    _SubGraph,
)

HIDDEN = 128


def _sub_graph():
    """Build a sub-graph on the relation model's hidden width."""
    torch.manual_seed(0)
    return _SubGraph(depth=3, hidden_size=HIDDEN)


@pytest.mark.integration
def test_layer_norm_standardises_every_row():
    """Layer norm centres each row and scales it to unit variance."""
    norm = _LayerNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        out = norm(x)

    assert out.shape == (2, 4)
    assert out[0].mean() == pytest.approx(0.0, abs=1e-5)
    assert out[0].std(unbiased=False) == pytest.approx(1.0, abs=1e-4)
    assert torch.allclose(out[1], torch.zeros(4))


@pytest.mark.integration
def test_mlp_relu_widths():
    """The MLP keeps its input width by default and honours an explicit one."""
    same = _MLP(4)
    wider = _MLP(4, 8)

    assert same(torch.zeros(2, 4)).shape == (2, 4)
    assert wider(torch.zeros(2, 4)).shape == (2, 8)
    assert (wider(torch.zeros(2, 4)) >= 0).all()


@pytest.mark.integration
def test_sub_graph_pools_each_polyline_to_one_vector():
    """The sub-graph reduces every polyline of a batch to a single vector."""
    sub_graph = _sub_graph()
    hidden = torch.randn(2, 5, HIDDEN)

    pooled = sub_graph(hidden, [5, 3])
    unpooled = sub_graph(hidden)

    assert pooled.shape == (2, HIDDEN)
    assert unpooled.shape == (2, HIDDEN)
    assert torch.isfinite(pooled).all()
    # Masking one polyline leaves the full-length one beside it alone.
    assert torch.allclose(pooled[0], unpooled[0])


@pytest.mark.integration
def test_merge_tensors_zero_pads_to_the_longest_polyline():
    """Shorter polylines are padded with zeros to the longest one."""
    short = torch.ones(2, HIDDEN)
    long = torch.ones(4, HIDDEN)

    merged, lengths = _merge_tensors([short, long], "cpu")

    assert lengths == [2, 4]
    assert merged.shape == (2, 4, HIDDEN)
    assert torch.allclose(merged[0, 2:], torch.zeros(2, HIDDEN))
    assert torch.allclose(merged[1], long)


@pytest.mark.integration
def test_merge_sub_graph_encodes_every_sample():
    """Each sample is merged and encoded separately."""
    sub_graph = _sub_graph()
    sample = [torch.randn(3, HIDDEN), torch.randn(5, HIDDEN)]

    single = _merge_sub_graph([sample], sub_graph, "cpu")
    double = _merge_sub_graph([sample, [torch.randn(2, HIDDEN)]], sub_graph, "cpu")

    assert len(single) == 1 and single[0].shape == (2, HIDDEN)
    assert [tensor.shape for tensor in double] == [(2, HIDDEN), (1, HIDDEN)]


@pytest.mark.integration
def test_global_graph_attends_over_the_unmasked_polylines():
    """The attention mask restricts which polylines a query may read."""
    graph = _GlobalGraph(HIDDEN, HIDDEN // 2)
    hidden = torch.randn(1, 3, HIDDEN)
    mask = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])

    unmasked = graph(hidden)
    masked = graph(hidden, mask)

    assert unmasked.shape == masked.shape == (1, 3, HIDDEN // 2)
    assert torch.isfinite(masked).all()
    assert not torch.allclose(unmasked, masked)
    # Without an explicit head size the graph attends at the hidden width.
    assert _GlobalGraph(HIDDEN)(hidden).shape == (1, 3, HIDDEN)


@pytest.mark.integration
def test_global_graph_res_doubles_the_width():
    """The residual global graph concatenates two attention heads."""
    graph = _GlobalGraphRes(HIDDEN)

    out = graph(torch.randn(1, 3, HIDDEN))

    assert out.shape == (1, 3, HIDDEN)


@pytest.mark.integration
def test_cross_attention_answers_every_query():
    """Cross attention returns one value per query row."""
    attention = _CrossAttention(HIDDEN)

    out = attention(torch.randn(1, 2, HIDDEN), torch.randn(1, 3, HIDDEN))

    assert out.shape == (1, 2, HIDDEN)
    assert torch.isfinite(out).all()


@pytest.mark.integration
def test_decoder_res_cat_scores_two_classes():
    """The decoder maps a pair vector to two class scores."""
    decoder = _DecoderResCat(in_features=HIDDEN * 2)

    out = decoder(torch.randn(1, HIDDEN * 2))

    assert out.shape == (1, 2)


def _pair_inputs(agent_polylines=4, road_polylines=2, rows=3):
    """Build a feature matrix with its polyline spans for one pair."""
    spans = [
        slice(index * rows, (index + 1) * rows) for index in range(agent_polylines + road_polylines)
    ]
    matrix = np.random.default_rng(0).normal(size=(len(spans) * rows, HIDDEN)).astype(np.float32)
    return matrix, spans, agent_polylines


@pytest.mark.integration
def test_relation_vector_net_scores_one_pair():
    """A pair scores as a two-class distribution over influencer and reactor."""
    model = RelationVectorNet()
    matrix, spans, map_start = _pair_inputs()

    scores = model(matrix, spans, map_start, "cpu")

    assert scores.shape == (1, 2)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all()
    assert scores.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.integration
def test_relation_vector_net_is_deterministic():
    """The same pair scores the same twice."""
    model = RelationVectorNet()
    matrix, spans, map_start = _pair_inputs()

    assert np.allclose(
        model(matrix, spans, map_start, "cpu"), model(matrix, spans, map_start, "cpu")
    )


@pytest.mark.integration
def test_from_checkpoint_loads_a_matching_state_dict(tmp_path):
    """A checkpoint holding every parameter reproduces the loaded model's scores."""
    model = RelationVectorNet()
    checkpoint = tmp_path / "relation.bin"
    torch.save(model.state_dict(), checkpoint)

    loaded = RelationVectorNet.from_checkpoint(str(checkpoint))
    matrix, spans, map_start = _pair_inputs()

    assert not loaded.training
    assert np.allclose(
        loaded(matrix, spans, map_start, "cpu"), model(matrix, spans, map_start, "cpu")
    )


@pytest.mark.integration
def test_from_checkpoint_rejects_an_incomplete_state_dict(tmp_path):
    """A checkpoint missing a parameter is refused rather than silently loaded."""
    state_dict = RelationVectorNet().state_dict()
    state_dict.pop(next(iter(state_dict)))
    checkpoint = tmp_path / "incomplete.bin"
    torch.save(state_dict, checkpoint)

    with pytest.raises(ValueError, match="missing parameters"):
        RelationVectorNet.from_checkpoint(str(checkpoint))


@pytest.mark.integration
def test_load_model_caches_each_checkpoint_path(tmp_path, monkeypatch):
    """Two configured checkpoints give two models, and one path loads once."""
    from tactics2d.behavior.intersim import relation_decider
    from tactics2d.behavior.intersim.config import InterSimConfig

    monkeypatch.setattr(relation_decider, "_MODEL_CACHE", {})
    paths = []
    for name in ("first.bin", "second.bin"):
        path = tmp_path / name
        torch.save(RelationVectorNet().state_dict(), path)
        paths.append(path)

    def load(path):
        return relation_decider.load_model(InterSimConfig(relation_model_path=str(path)))

    first = load(paths[0])

    assert load(paths[0]) is first
    assert load(paths[1]) is not first


@pytest.mark.integration
def test_load_model_requires_a_checkpoint_path():
    """A relation model without a configured checkpoint is a caller error."""
    from tactics2d.behavior.intersim import relation_decider
    from tactics2d.behavior.intersim.config import InterSimConfig

    with pytest.raises(ValueError, match="relation_model_path"):
        relation_decider.load_model(InterSimConfig())
