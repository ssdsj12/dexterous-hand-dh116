"""Tripod worker identity gates; CPU structural checks are not native leases."""
from copy import deepcopy
import json

import pytest

from test_native_teacher_worker_contract import recipe
from test_native_teacher_worker_semantic import config as semantic_config, binding
from test_native_teacher_tripod_admission import ROOT, MANIFEST, MANIFEST_SHA
from test_native_teacher_context import backend_profile


def test_tripod_recipe_requires_its_complete_source_closure_and_keeps_finite_budget(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
    spec=recipe(tmp_path)
    spec["teacher_contract"]["style"]="tripod"
    # Structural test supplies every current source module, then omits the
    # independent metric to prove that base Pinch closure cannot stand in for it.
    extras={str(p.relative_to(ROOT)) for p in (ROOT/"src/go2_pvcnn/control/dh116").glob("*.py")}
    spec["source_files"]=sorted(set(spec["source_files"])|extras)
    result=worker.validate_teacher_recipe_v2(spec)
    assert result["teacher_contract"]["style"]=="tripod"
    assert result["budget"]["updates"]==1 and result["budget"]["physics_steps"]==258
    bad=deepcopy(spec)
    bad["source_files"].remove("src/go2_pvcnn/control/dh116/tabletop_native_teacher_policy_tripod_distance.py")
    with pytest.raises(ValueError,match="source|closure"):
        worker.validate_teacher_recipe_v2(bad)
    for style in ("power","pinch"):
        bad=deepcopy(spec);bad["teacher_contract"]["style"]=style
        if style=="power":
            with pytest.raises(ValueError):worker.validate_teacher_recipe_v2(bad)
        else:
            assert worker.validate_teacher_recipe_v2(bad)["teacher_contract"]["style"]=="pinch"


def test_tripod_semantics_bind_independent_manifest_and_reject_cross_style_labels(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
    cfg,kw=semantic_config(tmp_path)
    cfg["preflight"].update(style="tripod",resolved_reset_manifest=dict(path=str(MANIFEST),sha256=MANIFEST_SHA))
    cfg["model_config"]=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_tripod_v1.json").read_bytes())
    good=worker.training_semantic_config(cfg,**kw)
    assert good["preflight"]["style"]==good["model_config"]["grasp"]=="tripod"
    assert good["preflight"]["resolved_reset_manifest"]==dict(sha256=MANIFEST_SHA)
    bad=deepcopy(cfg);bad["model_config"]["grasp"]="pinch"
    with pytest.raises(ValueError,match="model"):
        worker.training_semantic_config(bad,**kw)
    bad=deepcopy(cfg)
    path=tmp_path/"mislabeled-reset.json"
    data=json.loads(MANIFEST.read_bytes());data["style"]="pinch"
    path.write_text(json.dumps(data));bad["preflight"]["resolved_reset_manifest"]=binding(path)
    with pytest.raises(ValueError,match="Tripod|style|manifest"):
        worker.training_semantic_config(bad,**kw)


def test_checkpoint_target_accepts_actual_tripod_manifest_without_relabeling_materials(backend_profile):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import TripodResetManifest
    from test_native_teacher_tripod_admission import config
    manifest=load_reset_manifest(config(),project=ROOT)
    assert type(manifest) is TripodResetManifest
    target=worker.worker_training_target_contract(backend_profile,manifest)
    assert target==dict(target_schema="fixed_material_nodes41_set_distance_v1",target_material_ids=list(range(41)))
    assert manifest.data()["style"]=="tripod"
    with pytest.raises(ValueError):
        worker.worker_training_target_contract(backend_profile,manifest.data())
