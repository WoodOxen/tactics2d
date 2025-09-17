##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_trajgen.py
# @Description: This script is used to test the reproduction of TrajGen.
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np
import pytest

from tactics2d.behavior.trajgen.prioritized_replay_buffer import SumTree


@pytest.mark.behavior
def test_add_and_get_leaf_mapping():
    st = SumTree(capacity=4)
    items = ["a", "b", "c", "d"]
    priorities = [1, 2, 3, 4]

    for p, d in zip(priorities, items):
        st.add(p, d)

    samples = [(0.5, 1, "a"), (1.5, 2, "b"), (3.5, 3, "c"), (7.0, 4, "d")]

    for v, exp_p, exp_d in samples:
        _, pr, data = st.get_leaf(v)
        assert pr == exp_p
        assert data == exp_d


@pytest.mark.behavior
def test_update_changes_distribution():
    st = SumTree(capacity=3)
    items = ["x", "y", "z"]
    priorities = [1, 1, 1]

    for p, d in zip(priorities, items):
        st.add(p, d)

    idx_y, pr_y, data_y = st.get_leaf(1.5)
    assert data_y == "y"
    assert pr_y == 1

    st.update(idx_y, 20)

    _, pr, data = st.get_leaf(10.0)
    assert data == "y"
    assert pr == 20

    _, pr, data = st.get_leaf(0.5)
    assert data == "x"
    assert pr == 1

    _, pr, data = st.get_leaf(21.5)
    assert data == "z"
    assert pr == 1


@pytest.mark.behavior
def test_update_zero_excludes_leaf():
    st = SumTree(4)
    for p, d in zip([1, 1, 1, 1], list("abcd")):
        st.add(p, d)
    # 把 'b' 对应叶子设置为 0
    # 简便起见：先找到 'b'
    for i in range(st.capacity):
        if st.data[i] == "b":
            leaf = (st.capacity - 1) + i
            break
    st.update(leaf, 0.0)

    # Monte Carlo 采样确认几乎不命中 'b'
    hits = {k: 0 for k in "abcd"}
    T = 2000
    for _ in range(T):
        v = np.random.random() * max(st.total_priority, 1e-12)
        _, pr, data = st.get_leaf(v)
        hits[data] += 1
    assert hits["b"] == 0


@pytest.mark.behavior
def test_sampling_is_proportional():
    st = SumTree(5)
    pri = [1, 2, 3, 4, 10]
    for p, d in zip(pri, range(len(pri))):
        st.add(p, d)

    counts = np.zeros(len(pri), dtype=int)
    T = 20000
    for _ in range(T):
        v = np.random.random() * st.total_priority
        _, pr, data = st.get_leaf(v)
        counts[data] += 1

    probs = counts / counts.sum()
    target = np.array(pri) / sum(pri)
    # 宽松容差（卡方/LLN）；这里用绝对误差
    assert np.allclose(probs, target, atol=0.02)


@pytest.mark.behavior
def test_padding_leaves_never_sampled_when_leaf_count_upscaled():
    st = SumTree(3)  # 实现里 leaf_count=4, 末尾一片是 padding=0
    st.add(1, "x")
    st.add(1, "y")
    st.add(1, "z")
    T = 5000
    seen = set()
    for _ in range(T):
        v = np.random.random() * st.total_priority
        _, pr, data = st.get_leaf(v)
        seen.add(data)
    assert seen == {"x", "y", "z"}  # 不应出现 None/0/padding
