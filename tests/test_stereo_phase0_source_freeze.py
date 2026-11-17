"""Source provenance checks; no native imports or environment mutation."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase0_critical_source_freeze_matches_exact_audited_bytes():
    manifest = json.loads((ROOT / 'docs/contracts/stereo_phase0_source_manifest.json').read_text())
    assert manifest['base_commit'] == '6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b'
    assert manifest['instruction_lines'] == 930
    assert manifest['native_validation_performed'] is False
    assert len(manifest['critical_files']) == 21
    for row in manifest['critical_files']:
        content = (ROOT / row['path']).read_bytes()
        assert len(content) == row['bytes'], row['path']
        assert hashlib.sha256(content).hexdigest() == row['sha256'], row['path']
