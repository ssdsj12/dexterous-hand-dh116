"""Style-specific Worker auditing; synthetic CPU data never grants native status."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import PurePosixPath

import pytest

from test_native_teacher_context import backend_profile
from test_native_teacher_model import raw_cpu_packets
from test_native_teacher_actual_inputs import recorded_ledger
from test_native_teacher_tripod_admission import RUN
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,_digest


def tripod_ledger(profile):
    pre,raw,parameters,context=raw_cpu_packets(profile)
    actual=decode_owned(json.loads((RUN/"native/episode_reset.json").read_bytes()))["reset"]["rod_reset"]
    raw["mesh_names"]=deepcopy(actual["mesh_names"])
    raw["mesh_bindings"]=deepcopy(actual["mesh_bindings"])
    # CPU common fixture uses env8. Rebind only that declared fixture namespace,
    # never alter the historical native receipt or claim its measured poses.
    for name,row in raw["mesh_bindings"].items():
        row["prim_path"]=str(PurePosixPath("/World/envs/env_8")/(row["side"]+"Hand")/row["body_name"])
    context=replace(context,metadata=replace(context.metadata,grasp="tripod"))
    return recorded_ledger(profile,(pre,raw,parameters,context))


def audit_input(rows,profile,grasp="tripod"):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_inputs import audit_teacher_input_sources
    raw="".join(json.dumps(x,allow_nan=False)+"\n" for x in rows).encode()
    return audit_teacher_input_sources(raw,backend_profile=profile,scope="cpu_fixture",controls=1,grasp=grasp)


def test_tripod_decision_and_bootstrap_reconstruct_from_its_full_layout(backend_profile):
    rows=tripod_ledger(backend_profile)
    result=audit_input(rows,backend_profile)
    assert result["decision_packs"]==result["bootstrap_packs"]==1
    assert result["native_qualification"] is result["teacher_admission"] is False
    with pytest.raises(ValueError,match="grasp|context"):
        audit_input(rows,backend_profile,grasp="pinch")


@pytest.mark.parametrize("fault",["context","middle_missing","layout","side"])
def test_tripod_join_refuses_relabelled_context_and_hidden_midstep_layout_changes(backend_profile,fault):
    rows=tripod_ledger(backend_profile)
    if fault=="context":
        pack=next(x["payload"] for x in rows if x["operation"]=="teacher_input_source")
        pack["buffers"]["context"]["metadata"]["grasp"]="pinch"
        pack["sha256"]["context"]=_digest(decode_owned(pack["buffers"]["context"]))
    else:
        # Deliberately alter a middle SDK sample, not either Actor endpoint.
        rod=[x for x in rows if x["operation"]=="physical"][3]["payload"]["exchange"]["rod"]
        if fault=="middle_missing":del rod["mesh_bindings"]["right.finger33_link"]
        elif fault=="layout":rod["mesh_names"][-1],rod["mesh_names"][-2]=rod["mesh_names"][-2],rod["mesh_names"][-1]
        else:rod["mesh_bindings"]["right.finger33_link"]["side"]="left"
    with pytest.raises(ValueError,match="Tripod|grasp|context"):
        audit_input(rows,backend_profile)


def test_tripod_journal_requires_its_own_distance_operation_and_lossless_views(tmp_path,monkeypatch):
    from test_native_teacher_worker_artifacts import artifact_tree,write_jsonl
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
    spec,out,rows,result=artifact_tree(tmp_path)
    spec["teacher_contract"]["style"]="tripod"
    root=PurePosixPath(__file__).parents[1]
    from pathlib import Path
    extras={str(p.relative_to(Path(root))) for p in (Path(root)/"src/go2_pvcnn/control/dh116").glob("*.py")}
    spec["source_files"]=sorted(set(spec["source_files"])|extras)
    for row in rows:
        if row["operation"]=="policy_pinch_distance":row["operation"]="policy_tripod_distance"
    write_jsonl(out/"training_events.jsonl",rows)
    monkeypatch.setattr(artifacts,"_required_native_source_scope",lambda:"cpu_fixture")
    good=artifacts.audit_teacher_artifacts_v2(spec,out)
    assert good["journal_rows"]==rows
    changed=deepcopy(rows)
    next(x for x in changed if x["operation"]=="policy_tripod_distance")["operation"]="policy_pinch_distance"
    write_jsonl(out/"training_events.jsonl",changed)
    with pytest.raises(ValueError,match="sequence|journal"):
        artifacts.audit_teacher_artifacts_v2(spec,out)
