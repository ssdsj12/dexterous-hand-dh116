"""Recipe must lock protected assets as inputs, never publish them as source edits."""
import json
from pathlib import Path
from scripts.run_dh116_work_packages import release_path
ROOT=Path(__file__).resolve().parents[1]

def test_complete_pair_source_paths_are_scheduler_publishable():
    recipe=json.loads((ROOT/"configs/dh116_execution/complete_pair_r0_native_v1.json").read_text())
    for name in recipe["source_files"]:
        assert release_path(name)==name

def test_original_asset_locks_and_budget_are_unchanged():
    old=json.loads((ROOT/"configs/dh116_execution/fast_r0_reset_native_v1.json").read_text())
    new=json.loads((ROOT/"configs/dh116_execution/complete_pair_r0_native_v1.json").read_text())
    assert new["inputs"]==old["inputs"]
    assert new["budget"]==old["budget"]
    assert new["dependency_receipt"]==old["dependency_receipt"]
    assert set(old["source_files"])<=set(new["source_files"])
