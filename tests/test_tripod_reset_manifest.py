"""Tripod manifests derive state from the real exited17-slot Reset.

This is CPU replay, not a new native run or grasp qualification.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/fresh-supported-tripod-reset-native-v2")


def module():
    name = "go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest"
    assert importlib.util.find_spec(name) is not None, "independent Tripod17 Reset manifest reader missing"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def native_manifest(tmp_path_factory):
    cache = []
    def get():
        if not cache:
            m = module()
            output = tmp_path_factory.mktemp("tripod-manifest")/"manifest.json"
            outer = RUN/"outer.json"
            cache.append(m.seal_manifest(RUN,receipt=outer,expected_sha256=hashlib.sha256(outer.read_bytes()).hexdigest(),
                                         destination=output,project=ROOT))
        return cache[0]
    return get


def test_manifest_recovers_real_supported_reset_state_and_native_scene_inputs(native_manifest,monkeypatch):
    m = module()
    native_manifest = native_manifest()
    inventory={RUN/'native'/name for name in json.loads((RUN/'outer.json').read_bytes())['artifact_sha256']}
    # Each independent verification must parse exactly the bytes whose SHA
    # was checked, not a later unchecked read of the same artifact pathname.
    watched={RUN/'native'/name for name in {*m.FILES.values(),'frozen_plan.json',
        'result.json','episode_reset.json','cleanup_stages.json','app_close_returned.json'}}
    checked=m.checked_file
    read_bytes=Path.read_bytes
    active=set()
    reads={}
    def verified(path,expected):
        path=Path(path)
        active.add(path)
        try:
            result=checked(path,expected)
            if path in inventory:reads[path]=reads.get(path,0)+1
            return result
        finally:
            active.remove(path)
    def read_only_during_verification(path):
        if path in watched and path not in active:
            raise AssertionError('artifact reread after SHA verification: '+str(path))
        return read_bytes(path)
    monkeypatch.setattr(m,'checked_file',verified)
    monkeypatch.setattr(Path,'read_bytes',read_only_during_verification)
    config = dict(style="tripod",resolved_reset_manifest=native_manifest)
    loaded = m.load_tripod_reset_manifest(config,project=ROOT)
    assert all(reads.get(path,0)==1 for path in inventory)
    data = loaded.data()
    assert data["style"] == "tripod" and data["target_material_ids"] == list(range(41))
    assert data["reset_seed"]["sha256"] == "5b040a43753b43d2a02136e9c44016b8125c02b5b274fa766a563f2f68ba93dd"
    assert data["task_config"]["sha256"] == "6a635f5fd9d7d0415975cfa6de7192a3d632dc5eb5f374ac2634f3fca68898b9"
    assert len(data["rod_state"]["positions"][0]) == 41
    assert set(data["extra_sources"]) == {"rays","left_bake","right_bake","proximal_capture","proximal_bake"}
    assert data["scope"] == "native_reset_only_no_contact_or_grasp_admission"
    assert data["teacher_admission"] is data["grasp_qualified"] is False
    assert loaded.verify(project=ROOT) == loaded
    # load and verify both recheck all native files; there is no cross-call cache.
    assert all(reads.get(path,0)==2 for path in inventory)
    data["rod_state"]["positions"][0][0][0] += 1
    assert loaded.data()["rod_state"]["positions"][0][0][0] != data["rod_state"]["positions"][0][0][0]


@pytest.mark.parametrize("damage",["rod_position","task_identity","style"])
def test_rehashed_candidate_fields_cannot_replace_actual_tripod_native_sources(native_manifest,tmp_path,damage):
    m = module()
    native_manifest = native_manifest()
    data = json.loads(Path(native_manifest["path"]).read_bytes())
    if damage == "rod_position":
        data["rod_state"]["positions"][0][0][0] += .001
    elif damage == "task_identity":
        data["task_config"]["sha256"] = "a"*64
    else:
        data["style"] = "pinch"
    path=tmp_path/"rehashed.json"
    path.write_text(json.dumps(data))
    binding=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="Tripod"):
        m.load_tripod_reset_manifest(dict(style="tripod",resolved_reset_manifest=binding),project=ROOT)
