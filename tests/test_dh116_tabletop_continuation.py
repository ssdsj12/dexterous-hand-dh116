"""Real subprocess orchestration tests with an explicitly synthetic trainer."""
import json
import sys
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_continuation import continue_training


def fake_trainer(tmp_path, outcomes):
    script = tmp_path/'fake_trainer.py'
    script.write_text('''import json,sys,time
from pathlib import Path
out=Path(sys.argv[1]); out.mkdir()
resume=Path(sys.argv[2]); payload=json.loads(sys.argv[3])
assert resume.exists()
(out/'ppo_checkpoint.zip').write_text('synthetic model')
(out/'ppo_checkpoint.json').write_text(json.dumps({'stage':'close','timesteps':payload.get('steps',128)}))
(out/'summary.json').write_text(json.dumps(payload))
sys.exit(payload.get('process_exit_code',0))
''')
    seed = tmp_path/'seed.zip';seed.write_text('synthetic model')
    def command(checkpoint, output, index):
        return [sys.executable, str(script), str(output), str(checkpoint), json.dumps(outcomes[index-1])]
    return seed, command


def test_safe_unsolved_budget_automatically_resumes_then_stops_on_qualified_grasp(tmp_path):
    seed, command = fake_trainer(tmp_path, [
        dict(training_stopped='curriculum_stage_not_mastered', passed=False, grasp_qualified=False),
        dict(passed=True, grasp_qualified=True)])
    out = tmp_path/'run'
    result = continue_training(seed, out, command, max_rounds=5, poll_interval=.01)
    assert result['status'] == 'qualified'
    assert len(result['rounds']) == 2
    assert result['rounds'][1]['resumed_from'] == str(out/'round_0001/ppo_checkpoint.zip')
    assert not (out/'round_0003').exists()


def test_physical_failure_never_automatically_restarts(tmp_path):
    seed, command = fake_trainer(tmp_path, [dict(
        training_stopped='physical_safety_failure', process_exit_code=2)])
    result = continue_training(seed, tmp_path/'run', command, max_rounds=5, poll_interval=.01)
    assert result['status'] == 'failed'
    assert len(result['rounds']) == 1
    assert result['rounds'][0]['exit_code'] == 2


def test_budget_exhaustion_is_not_a_success_claim(tmp_path):
    seed, command = fake_trainer(tmp_path, [dict(training_stopped='curriculum_stage_not_mastered')]*2)
    result = continue_training(seed, tmp_path/'run', command, max_rounds=2, poll_interval=.01)
    assert result['status'] == 'budget_exhausted'
    assert not result['grasp_qualified']
    assert Path(result['checkpoint']).is_file()


def test_runtime_code_change_between_rounds_prevents_silent_hot_reload(tmp_path):
    seed, build = fake_trainer(tmp_path, [dict(training_stopped='curriculum_stage_not_mastered')])
    runtime = tmp_path/'runtime.py';runtime.write_text('before')
    def command(*args):
        runtime.write_text('changed')
        return build(*args)
    result = continue_training(seed, tmp_path/'run', command, max_rounds=2,
                               source_paths=[runtime], poll_interval=.01)
    assert result['status'] == 'source_changed'
    assert len(result['rounds']) == 1
