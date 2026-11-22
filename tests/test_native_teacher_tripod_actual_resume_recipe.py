"""Historical v11 resume cannot silently reuse a source-changed physical exchange."""
from pathlib import Path
import pytest
from scripts.run_dh116_work_packages import Repository

ROOT = Path(__file__).resolve().parents[1]


def test_historical_tripod_resume_rejects_changed_exchange_before_native_launch():
    repo = Repository(ROOT, evidence_dir=Path.home()/".dh116-native-control-11-17")
    packages = repo.registry()["packages"]
    resume = packages["teacher.tripod-ppo-resume-smoke8-v11"]
    assert resume["command_spec"] == packages["teacher.tripod-ppo-fresh-smoke8-v11"]["command_spec"]
    with pytest.raises(ValueError, match="v2 source inventory SHA differs from actual scheduler source closure"):
        repo.native_recipe(resume)
