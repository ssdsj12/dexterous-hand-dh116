"""CPU semantic identity and retained-receipt parsing, never native proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_contract as worker
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def binding(path):return dict(path=str(path),sha256=sha(path))

def config(tmp_path):
    pre=candidate_config()
    reward=ROOT/"configs/dh116_training/approach_wiring_smoke_v1.json"
    reset=tmp_path/"reset.json";reset.write_text(json.dumps({"target":"fixed41","seed":1}))
    pre.update(reward_config=binding(reward),resolved_reset_manifest=binding(reset),
        budget=dict(max_sdk_steps=258,max_resets=1,max_solver_calls=256,max_episode_count=1,wall_timeout_s=180))
    model=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_pinch_v1.json").read_text())
    cfg=dict(schema="dh116-native-teacher-training-cpu-v1",preflight=pre,model_config=model,episode_cap_s=20.,rollout_steps=32)
    backend={k:"1"*64 for k in ["profile_sha256","binary_sha256","inventory_sha256","source_tree_sha256","patch_sha256"]}
    sources={"source.py":"2"*64}
    physics=dict(physics_hz=480,control_hz=60,solver_substeps=12,physics_config_sha256="3"*64,
        action_contract_sha256="4"*64,target_schema="fixed_material_nodes41_set_distance_v1")
    return cfg,dict(backend_identity=backend,source_identity=sources,physics_identity=physics)


def test_process_lease_budget_and_binding_path_do_not_change_semantics(tmp_path):
    cfg,kw=config(tmp_path)
    assert callable(getattr(worker,"training_semantic_config",None)), "versioned training semantic contract missing"
    first=worker.training_semantic_digest(cfg,**kw)
    cfg["preflight"]["authorization"]={"path":"/private/process-A.json","sha256":"5"*64}
    cfg["preflight"]["resource_lock"]={"path":"/private/process-B.json","sha256":"6"*64}
    cfg["preflight"]["budget"]["wall_timeout_s"]=200
    previous=Path(cfg["preflight"]["resolved_reset_manifest"]["path"])
    other=tmp_path/"moved-reset.json";other.write_bytes(previous.read_bytes())
    cfg["preflight"]["resolved_reset_manifest"]=binding(other)
    assert worker.training_semantic_digest(cfg,**kw)==first
    semantic=worker.training_semantic_config(cfg,**kw)
    assert semantic["schema"]=="dh116-native-teacher-training-semantic-v1"
    assert semantic["policy_execution"]==dict(device="cpu",optimizer_device="cpu",rng_schema="torch_cpu_rng_v1")
    assert not {"authorization","resource_lock","budget"} & semantic["preflight"].keys()


@pytest.mark.parametrize("change",["model","ppo","episode_cap","backend","sources","physics","action","reset"])
def test_all_behavior_changes_change_semantic_identity(tmp_path,change):
    cfg,kw=config(tmp_path);first=worker.training_semantic_digest(cfg,**kw)
    if change=="model":cfg["model_config"]["model"]["initial_std"]=.2
    elif change=="ppo":cfg["preflight"]["ppo"]["lam"]=.9
    elif change=="episode_cap":cfg["episode_cap_s"]=19.
    elif change=="backend":kw["backend_identity"]["binary_sha256"]="9"*64
    elif change=="sources":kw["source_identity"]["source.py"]="9"*64
    elif change in ("physics","action"):kw["physics_identity"]["physics_config_sha256" if change=="physics" else "action_contract_sha256"]="9"*64
    else:
        p=Path(cfg["preflight"]["resolved_reset_manifest"]["path"]);p.write_text('{"target":"fixed41","seed":2}')
        cfg["preflight"]["resolved_reset_manifest"]=binding(p)
    assert worker.training_semantic_digest(cfg,**kw)!=first


@pytest.mark.parametrize("damage",["unknown_outer","unknown_preflight","unknown_model","bad_sha","tampered_file","gamma","unknown_physics","nonfinite"])
def test_semantics_are_validated_before_hashing(tmp_path,damage):
    cfg,kw=config(tmp_path)
    if damage=="unknown_outer":cfg["output"]=str(tmp_path)
    elif damage=="unknown_preflight":cfg["preflight"]["new_behavior"]=1
    elif damage=="unknown_model":cfg["model_config"]["model"]["new_behavior"]=1
    elif damage=="bad_sha":kw["source_identity"]["source.py"]="bad"
    elif damage=="tampered_file":Path(cfg["preflight"]["resolved_reset_manifest"]["path"]).write_text("changed")
    elif damage=="gamma":cfg["preflight"]["ppo"]["gamma"]=.9
    elif damage=="unknown_physics":kw["physics_identity"]["new_behavior"]=1
    else:cfg["episode_cap_s"]=float("inf")
    with pytest.raises(ValueError):worker.training_semantic_digest(cfg,**kw)


def receipt_tree(tmp_path):
    output=tmp_path/"native";output.mkdir();ck=output/"teacher_clean.pt";ck.write_bytes(b"synthetic CPU checkpoint bytes")
    result=dict(schema=worker.RESULT_SCHEMA,semantic_sha256="1"*64,
        clean_checkpoint=dict(path="teacher_clean.pt",sha256=sha(ck)),
        training=dict(loaded_updates=0,target_updates=1,updates=1,accepted_minibatch_updates=1,parameter_delta_l2=.1))
    rp=output/"training_result.json";rp.write_text(json.dumps(result))
    receipt=dict(output=str(output),native_qualification=True,native_execution_performed=True,outer_exit_code=0,
        artifact_sha256={"training_result.json":sha(rp),"teacher_clean.pt":sha(ck)})
    path=tmp_path/"receipt.json";path.write_text(json.dumps(receipt))
    calls=[]
    class Repo:
        def registry(self):return {"packages":{"A":{"evidence_scope":"pinch-smoke","status":"passed","executor":"native_bounded"}}}
        def native_recipe(self,package):return {"spec":{"version":2,"teacher_contract":{"semantic_sha256":"1"*64},"validator":{"result_file":"training_result.json"}}}
        def _verified_native_receipt(self,*args):calls.append(args);return path
    return Repo(),path,rp,ck,calls


def test_b_checkpoint_is_resolved_only_from_reverified_a_bytes(tmp_path):
    repo,path,rp,ck,calls=receipt_tree(tmp_path)
    actual=worker.checkpoint_from_a_receipt(repo,"A",sha(path),expected_semantic_sha256="1"*64)
    assert calls and calls[0][0:2]==("A","pinch-smoke")
    assert actual==dict(path=str(ck),sha256=sha(ck),loaded_updates=1,semantic_sha256="1"*64)


@pytest.mark.parametrize("damage",["receipt_sha","result_sha","checkpoint_sha","semantic","failed","updates","zero_delta","escape","unqualified"])
def test_b_rejects_unqualified_or_changed_a(tmp_path,damage):
    repo,path,rp,ck,calls=receipt_tree(tmp_path);expected=sha(path)
    receipt=json.loads(path.read_text());result=json.loads(rp.read_text())
    if damage=="receipt_sha":path.write_text("{}")
    elif damage=="result_sha":rp.write_text("{}")
    elif damage=="checkpoint_sha":ck.write_bytes(b"changed")
    else:
        if damage=="semantic":result["semantic_sha256"]="2"*64
        elif damage=="failed":receipt["outer_exit_code"]=1
        elif damage=="updates":result["training"]["updates"]=2
        elif damage=="zero_delta":result["training"]["parameter_delta_l2"]=0.
        elif damage=="escape":result["clean_checkpoint"]["path"]="../teacher_clean.pt"
        elif damage=="unqualified":receipt["native_qualification"]=False
        rp.write_text(json.dumps(result));receipt["artifact_sha256"]["training_result.json"]=sha(rp)
        path.write_text(json.dumps(receipt));expected=sha(path)
    with pytest.raises(ValueError):worker.checkpoint_from_a_receipt(repo,"A",expected,expected_semantic_sha256="1"*64)
