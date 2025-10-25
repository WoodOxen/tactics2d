##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_trajgen.py
# @Description: This script is used to test the reproduction of TrajGen.
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np
import pytest

from tactics2d.behavior.trajgen.prioritized_replay_buffer import PrioritizedReplayBuffer, SumTree


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
    for i in range(st.capacity):
        if st.data[i] == "b":
            leaf = (st.capacity - 1) + i
            break
    st.update(leaf, 0.0)

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

    assert np.allclose(probs, target, atol=0.02)


@pytest.mark.behavior
def test_padding_leaves_never_sampled_when_leaf_count_upscaled():
    st = SumTree(3)
    st.add(1, "x")
    st.add(1, "y")
    st.add(1, "z")
    T = 5000
    seen = set()
    for _ in range(T):
        v = np.random.random() * st.total_priority
        _, pr, data = st.get_leaf(v)
        seen.add(data)
    assert seen == {"x", "y", "z"}


@pytest.fixture
def small_buffer():
    return PrioritizedReplayBuffer(buffer_size=8, alpha=0.6, beta=0.4, beta_increment=1e-3)


@pytest.mark.behavior
def test_push_increases_total_priority(small_buffer):
    buf = small_buffer
    # push 使用当前最大叶子优先级（首个为 upper_absolute_error）
    k = 5
    for i in range(k):
        buf.push(("s", i))

    total = buf.sum_tree.total_priority
    assert np.isclose(total, k * buf.upper_absolute_error)

    # 前 k 条 data 不为默认 0.0
    assert all(buf.sum_tree.data[i] != 0.0 for i in range(k))


@pytest.mark.behavior
def test_sample_shapes_and_ids_and_data(small_buffer):
    buf = small_buffer
    data = [("exp", i) for i in range(6)]
    for d in data:
        buf.push(d)

    n = 4
    ids, exps, w = buf.sample(n)

    assert ids.shape == (n,)
    assert w.shape == (n, 1)  # 你当前实现是 (n,1)
    assert len(exps) == n

    # 采样的树索引必须在叶子区间内
    start = buf.sum_tree.leaf_start
    end = start + buf.buffer_size
    assert np.all((ids >= start) & (ids < end))

    # 经验必须来自已写入的数据槽（而不是初始化的 0.0）
    for e in exps:
        assert isinstance(e, tuple) and e[0] == "exp"


@pytest.mark.behavior
def test_update_priority_changes_leaf_value(small_buffer):
    buf = small_buffer
    for i in range(3):
        buf.push(("x", i))

    ids, exps, w = buf.sample(2)  # 抽两个
    # 构造一个较大的 TD 误差，观察优先级变化
    abs_errors = np.array([2.5, 0.1], dtype=float)
    buf.update_priority(ids[:2], abs_errors[:2])

    # 逐个检查叶子值是否等于 min(|e|+eps, cap)^alpha
    for idx, e in zip(ids[:2], abs_errors[:2]):
        expected = min(e + buf.eps, buf.upper_absolute_error) ** buf.alpha
        assert np.isclose(buf.sum_tree.tree[int(idx)], expected)


@pytest.mark.behavior
def test_importance_weights_are_normalized(small_buffer):
    buf = small_buffer
    for i in range(6):
        buf.push(("x", i))

    ids, exps, w = buf.sample(6)
    # 归一化后 max_weight 应当接近 1（数值误差允许 1.000... 一点点偏差）
    assert w.max() <= 1.0 + 1e-6
    assert w.min() > 0.0


@pytest.mark.behavior
def test_sampling_raises_on_empty_buffer():
    buf = PrioritizedReplayBuffer(4)
    with pytest.raises(ValueError):
        buf.sample(2)
