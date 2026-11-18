import pytest

from go2_pvcnn.control.dh116.runtime_snapshot import ReactionBuffer, SnapshotStore


def test_pre_and_post_snapshots_have_distinct_tokens():
    store = SnapshotStore()
    pre = store.capture_pre(step=3, stage="approach", sample_time=.0003, values={"q": [1]})
    post = store.capture_post(step=3, stage="approach", sample_time=.0003, values={"q": [2]})
    assert pre.token != post.token
    assert pre.values["q"] == [1]
    assert post.values["q"] == [2]
    with pytest.raises(TypeError):
        pre.values["q"].append(4)
    with pytest.raises(TypeError):
        pre.values["q"] = [3]


def test_reaction_buffer_clears_in_place():
    buffer = ReactionBuffer.allocate("cpu", 4)
    identity = id(buffer.force)
    buffer.force.fill_(1.0)
    buffer.torque.fill_(1.0)
    buffer.clear_()
    assert id(buffer.force) == identity
    assert bool((buffer.force == 0).all())
    assert bool((buffer.torque == 0).all())
