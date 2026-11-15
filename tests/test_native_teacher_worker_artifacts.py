"""Independent v2 scheduler regressions, synthetic CPU ledgers, no native proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
from test_native_teacher_worker_contract import recipe, pin_source_closure


@pytest.fixture(autouse=True)
def cpu_native_source_gate(monkeypatch):
    """Only lower native-source scope is substituted; rows retain CPU labels.

    All actual row/source/metadata/policy/checkpoint auditing remains enabled.
    This isolated software fixture never supplies native execution evidence.
    """
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
    original=artifacts._required_native_source_scope
    monkeypatch.setattr(artifacts,"_required_native_source_scope",lambda:"cpu_fixture")
    return original


def write_jsonl(path,rows):path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def artifact_tree(tmp_path):
    spec=recipe(tmp_path);out=tmp_path/"native";out.mkdir()
    rows=[];physical=[];transitions=[];ownership=[]
    def emit(operation,payload):
        index=len(rows);rows.append(dict(sequence=index,scope="cpu_fixture",operation=operation,payload=payload));return dict(raw_sequence=index,payload=deepcopy(payload))
    origin=dict(session_id="synthetic-cpu-session",policy_version=0,weights="0"*64,model="8"*64)
    emit("collector_origin",dict(policy=origin,native_qualification=False,teacher_admission=False,scope="cpu_fixture"))
    for control in range(32):
        token=f"cpu-control-{control}"
        emit("decision",dict(policy=origin,collection_token=token))
        for substep in range(8):
            sdk=control*8+substep+3
            emit("delivered_exchange",dict(sdk_step=sdk))
            physical.append(emit("physical",dict(sdk_step=sdk)))
            emit("budget_consumed",dict(sdk_steps=sdk-2,solver_calls=sdk-2,internal_solver_substeps=(sdk-2)*12))
            emit("policy_pinch_distance",dict(sdk_step=sdk))
        emit("settlement",dict(reward=dict(token=token)))
        transitions.append(emit("transition_recorded",dict(collection_token=token,rollout_length=control+1)))
    emit("policy_epoch",dict(origin,policy_version=1,weights="1"*64))
    emit("optimizer_update_accepted",dict(updates=1,accepted_minibatch_updates=1,parameter_delta_l2=.1,metrics=dict(minibatch_updates=1)))
    ownership.append(emit("owner_lease_closed",dict(native_resources_released=False,borrowed_references_dropped=True,app_closed=False)))
    write_jsonl(out/"physics.jsonl",physical);write_jsonl(out/"transitions.jsonl",transitions)
    write_jsonl(out/"ownership.jsonl",ownership);write_jsonl(out/"training_events.jsonl",rows)
    ck=out/"teacher_clean.pt";ck.write_bytes(b"{}\n")
    result=dict(schema=worker.RESULT_SCHEMA,semantic_sha256=spec["teacher_contract"]["semantic_sha256"],
        training=dict(loaded_updates=0,target_updates=1,updates=1,accepted_minibatch_updates=1,parameter_delta_l2=.1),
        clean_checkpoint=dict(path="teacher_clean.pt",sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
    (out/"training_result.json").write_text(json.dumps(result))
    (out/"shutdown_complete.json").write_text(json.dumps(dict(close_returned=True,pending_simulation_callback_error=None,process_exit_verified=False)))
    return spec,out,rows,result


@pytest.mark.parametrize("ledger,maximum",[("ownership.jsonl",256),("training_events.jsonl",2048)])
def test_real_scheduler_enforces_every_v2_jsonl_upper_bound(tmp_path,ledger,maximum):
    spec,out,rows,result=artifact_tree(tmp_path)
    if ledger=="ownership.jsonl":
        values=[dict(raw_sequence=0,payload={}) for _ in range(maximum+1)]
    else:values=rows+[dict(sequence=i,scope="cpu_fixture",operation="owner_event",payload={}) for i in range(len(rows),maximum+1)]
    write_jsonl(out/ledger,values)
    with pytest.raises(ValueError,match="budget|bound|exceed"):
        scheduler.validate_native_artifacts(spec,out)


@pytest.mark.parametrize("damage",["updates","accepted","delta","semantic","missing_journal_acceptance"])
def test_real_scheduler_rejects_unperformed_or_unbound_update(tmp_path,damage):
    spec,out,rows,result=artifact_tree(tmp_path)
    if damage=="updates":result["training"]["updates"]=0
    elif damage=="accepted":result["training"]["accepted_minibatch_updates"]=0
    elif damage=="delta":result["training"]["parameter_delta_l2"]=0.
    elif damage=="semantic":result["semantic_sha256"]="9"*64
    else:
        rows=[row for row in rows if row["operation"]!="optimizer_update_accepted"]
        for i,row in enumerate(rows):row["sequence"]=i
        write_jsonl(out/"training_events.jsonl",rows)
    (out/"training_result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError,match="update|semantic|accepted|change|journal"):
        scheduler.validate_native_artifacts(spec,out)


def test_real_scheduler_cannot_qualify_hash_matching_text_as_clean_checkpoint(tmp_path):
    spec,out,rows,result=artifact_tree(tmp_path)
    with pytest.raises(ValueError,match="checkpoint"):
        scheduler.validate_native_artifacts(spec,out)


@pytest.mark.parametrize("ledger",["physics.jsonl","transitions.jsonl","ownership.jsonl","training_events.jsonl"])
@pytest.mark.parametrize("damage",["partial_line","non_object","nonfinite"])
def test_all_four_v2_streams_require_complete_finite_objects(tmp_path,ledger,damage):
    spec,out,rows,result=artifact_tree(tmp_path);path=out/ledger
    if damage=="partial_line":path.write_bytes(path.read_bytes().rstrip(b"\n"))
    elif damage=="non_object":path.write_bytes(b"[]\n")
    else:path.write_bytes(b'{"bad":NaN}\n')
    with pytest.raises(ValueError,match="newline|object|JSON|nonfinite"):
        scheduler.validate_native_artifacts(spec,out)


@pytest.mark.parametrize("damage",["duplicate_sequence","gap","sdk_operation_order","same_weights","different_session","accepted_metric","derived_reference","derived_payload","derived_number_type"])
def test_v2_journal_and_derived_views_stay_source_joined(tmp_path,damage):
    spec,out,rows,result=artifact_tree(tmp_path)
    if damage=="duplicate_sequence":rows[5]["sequence"]=rows[4]["sequence"]
    elif damage=="gap":rows[5]["sequence"]+=1
    elif damage=="sdk_operation_order":rows[2]["operation"],rows[3]["operation"]=rows[3]["operation"],rows[2]["operation"]
    elif damage=="same_weights":next(r for r in rows if r["operation"]=="policy_epoch")["payload"]["weights"]="0"*64
    elif damage=="different_session":next(r for r in rows if r["operation"]=="policy_epoch")["payload"]["session_id"]="other"
    elif damage=="accepted_metric":next(r for r in rows if r["operation"]=="optimizer_update_accepted")["payload"]["parameter_delta_l2"]=.2
    else:
        name="physics.jsonl";values=[json.loads(l) for l in (out/name).read_text().splitlines()]
        if damage=="derived_reference":values[0]["raw_sequence"]+=1
        elif damage=="derived_number_type":values[0]["payload"]["sdk_step"]=float(values[0]["payload"]["sdk_step"])
        else:values[0]["payload"]["sdk_step"]+=1
        write_jsonl(out/name,values)
    write_jsonl(out/"training_events.jsonl",rows)
    with pytest.raises(ValueError,match="sequence|journal|epoch|session|view|metric"):
        scheduler.validate_native_artifacts(spec,out)


def test_rows_alone_cannot_qualify_checkpoint_without_actual_repository(tmp_path):
    spec,out,rows,result=artifact_tree(tmp_path)
    # Full checkpoint audit requires an actual source-bound recipe context.
    with pytest.raises(ValueError,match="checkpoint.*Repository"):
        scheduler.validate_native_artifacts(spec,out)


def registered_context(tmp_path,spec):
    root=tmp_path/"registered-repo";root.mkdir()
    for name in [spec["entrypoint"],"scripts/python.sh","scripts/run_dh116_work_packages.py",*spec["source_files"]]:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("# isolated CPU context fixture\n")
    pin_source_closure(root,spec)
    name="configs/dh116_execution/cpu-audit-fixture.json";p=root/name;p.parent.mkdir(parents=True);p.write_text(json.dumps(spec))
    package=dict(status="pending",executor="native_bounded",title="CPU audit fixture",dependencies=[],tests=[],
        command_spec=name,requires_approval=True,resource_class=scheduler.NATIVE_SCOPE,evidence_scope="cpu_fixture",
        dependency_scopes={},retry_policy=scheduler.NATIVE_RETRY)
    path=root/scheduler.REGISTRY;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(version=2,packages={"native-a":package})))
    return scheduler.Repository(root),package


def test_v2_audit_context_uses_actual_registry_and_frozen_recipe(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
    assert callable(getattr(artifacts,"_current_recipe_context",None)),"source-bound registry audit context missing"
    spec,out,rows,result=artifact_tree(tmp_path);repo,package=registered_context(tmp_path,spec)
    context=artifacts._current_recipe_context(spec,repo,package)
    assert context["package_name"]=="native-a" and context["recipe"]["spec"]==spec
    assert context["source_contract"]=={"worker_source_closure":spec["teacher_contract"]["source_sha256"]}
    changed=deepcopy(package);changed["title"]="caller invented package"
    with pytest.raises(ValueError,match="registered|registry"):
        artifacts._current_recipe_context(spec,repo,changed)
    changed_spec=deepcopy(spec);changed_spec["teacher_contract"]["semantic_sha256"]="9"*64
    with pytest.raises(ValueError,match="frozen|recipe"):
        artifacts._current_recipe_context(changed_spec,repo,package)


def test_v2_qualification_rejects_missing_or_fake_repository_context(tmp_path):
    from types import SimpleNamespace
    spec,out,rows,result=artifact_tree(tmp_path)
    for repository in (None,SimpleNamespace(root=tmp_path)):
        with pytest.raises(ValueError,match="Repository|context"):
            scheduler.validate_native_artifacts(spec,out,repository=repository,package={})


def test_v1_artifacts_ignore_optional_v2_context_keywords(tmp_path):
    out=tmp_path/"v1";out.mkdir();(out/"result.json").write_text('{"ok":true}')
    (out/"rows.jsonl").write_text('{}\n')
    spec=dict(version=1,validator=dict(result_file="result.json",equals={"ok":True},
        artifacts=["result.json","rows.jsonl"],line_counts={"rows.jsonl":1}))
    assert scheduler.validate_native_artifacts(spec,out)==scheduler.validate_native_artifacts(spec,out,repository=object(),package=object())


@pytest.mark.parametrize("damage",["missing_training","training_not_object","settlement_not_object","accepted_metrics_missing"])
def test_malformed_update_metadata_is_an_explicit_audit_rejection(tmp_path,damage):
    spec,out,rows,result=artifact_tree(tmp_path)
    if damage=="missing_training":del result["training"]
    elif damage=="training_not_object":result["training"]=None
    elif damage=="settlement_not_object":next(r for r in rows if r["operation"]=="settlement")["payload"]["reward"]=None
    else:del next(r for r in rows if r["operation"]=="optimizer_update_accepted")["payload"]["metrics"]
    (out/"training_result.json").write_text(json.dumps(result));write_jsonl(out/"training_events.jsonl",rows)
    with pytest.raises(ValueError,match="training|settlement|accepted|metrics"):
        scheduler.validate_native_artifacts(spec,out)


def test_production_journal_rejects_cpu_fixture_scope(tmp_path,monkeypatch,cpu_native_source_gate):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as artifacts
    spec,out,rows,result=artifact_tree(tmp_path)
    monkeypatch.setattr(artifacts,"_required_native_source_scope",cpu_native_source_gate)
    assert artifacts._required_native_source_scope()=="owner_source_unqualified"
    with pytest.raises(ValueError,match="scope"):
        artifacts._journal(rows,result)
