import inspect
from scripts import replay_dh116_effective_pad_r0 as replay

def test_optional_reducer_preserves_legacy_replay_default():
    option=inspect.signature(replay.run).parameters["negative_reducer_cls"]
    assert option.default is None
