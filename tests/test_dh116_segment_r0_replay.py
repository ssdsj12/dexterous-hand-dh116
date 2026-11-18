import hashlib,json
import pytest

def script():
    from scripts import replay_dh116_segment_contact_r0 as s
    return s

def write(tmp_path,text):
    p=tmp_path/"e.jsonl";p.write_text(text)
    return p,hashlib.sha256(p.read_bytes()).hexdigest()

def test_stream_preserves_header_and_complete_records(tmp_path):
    p,h=write(tmp_path,'{"header":1}\n{"operation":"physical","payload":{"x":2}}\n')
    assert list(script().stream_events(p,h))==[{"header":1},{"operation":"physical","payload":{"x":2}}]

@pytest.mark.parametrize("text",['{"x":1}', '{"x":1}\n\n', '{"x":1}\nBAD\n'])
def test_stream_rejects_incomplete_or_invalid_rows(tmp_path,text):
    p,h=write(tmp_path,text)
    with pytest.raises(ValueError):list(script().stream_events(p,h))

def test_stream_refuses_changed_file_and_symlink(tmp_path):
    p,h=write(tmp_path,'{"x":1}\n');p.write_text('{"x":2}\n')
    with pytest.raises(ValueError):list(script().stream_events(p,h))
    q=tmp_path/"link";q.symlink_to(p)
    with pytest.raises(ValueError):list(script().stream_events(q,hashlib.sha256(p.read_bytes()).hexdigest()))

def test_pinned_source_modified_while_streaming_cannot_complete(tmp_path):
    p,h=write(tmp_path,'{"x":1}\n{"x":2}\n')
    g=script().stream_events(p,h);next(g);p.write_text('{"x":9}\n')
    with pytest.raises(ValueError):list(g)

def test_exact_observed_prefix_is_not_full_episode_or_training():
    s=script();r=s.prefix_scope(29,232,2)
    assert r["total_recorded_sdk_steps"]==234 and not r["full_episode_evaluated"]
    assert r["training_reward"] is None and not r["teacher_admission"]
    with pytest.raises(ValueError):s.prefix_scope(29,231,2)
    with pytest.raises(ValueError):s.prefix_scope(30,240,2)
