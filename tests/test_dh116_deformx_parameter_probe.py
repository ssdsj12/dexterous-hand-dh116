"""CPU checks of the zero-step getter experiment; no GPU qualification."""
import copy
import hashlib
import json
from pathlib import Path
import pytest

from scripts import verify_dh116_deformx_parameter_getters as probe
from tests.test_dh116_deformx_parameter_getters import packet

def evidence():
    raw = packet()
    raw["stretch_ea_n"][0, 1] = 1200.
    raw["stored_k_stretch_n_m"][0, 1] = .02 * 1200
    raw["bend_ei_n_m2"][0, 1] = .027
    raw["rest_quaternion_norm_n_m"][0, 1] = 4 * .027 / .02
    raw["inverse_masses_kg_inv"][0] = [1000, 500, 1/.003]
    plain = {k: v.tolist() if hasattr(v, "tolist") else v for k,v in raw.items()}
    state = {"positions": [[[0,0,1],[0,.01,1],[0,.03,1]]],
             "velocities": [[[0,0,0]]*3], "quaternions": [[[0,0,0,1]]*3]}
    return {"snapshots": [copy.deepcopy(plain) for _ in range(3)],
            "state_before": state, "state_after": copy.deepcopy(state),
            "time_before": 0., "time_after": 0., "solver_last_error": 0,
            "mutated_owned_host_arrays": sorted(k for k,v in raw.items() if hasattr(v,"shape"))}

def test_raw_reopening_checks_all_parameters_and_state(tmp_path):
    raw = evidence()
    result = probe.validate_raw(raw)
    assert result["cuda_solver_steps"] == 0 and result["snapshot_count"] == 3
    assert result["native_qualification_scope"] == "zero_step_parameter_getter_only"
    probe.write_new(tmp_path/"raw_parameters.json",raw)
    probe.write_new(tmp_path/"result.json",result)
    probe.validate_artifacts(tmp_path)
    bad = json.loads((tmp_path/"result.json").read_text());bad["snapshot_count"]=1
    (tmp_path/"result.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):probe.validate_artifacts(tmp_path)

@pytest.mark.parametrize("damage", ["rest","ea","ei","radius","position","velocity","clock","error","snapshot","mutations"])
def test_wrong_or_incomplete_raw_evidence_fails(damage):
    raw=evidence()
    if damage=="rest":raw["snapshots"][1]["rest_lengths_m"]=[[.02,.02]]
    if damage=="ea":raw["snapshots"][2]["stretch_ea_n"]=[[800,800]]
    if damage=="ei":raw["snapshots"][0]["bend_ei_n_m2"]=[[.018,.018]]
    if damage=="radius":raw["snapshots"][1]["contact_radius_m"]=.008
    if damage=="position":raw["state_after"]["positions"][0][0][2]+=.001
    if damage=="velocity":raw["state_after"]["velocities"][0][0][2]=1
    if damage=="clock":raw["time_after"]=1/480
    if damage=="error":raw["solver_last_error"]=1
    if damage=="snapshot":raw["snapshots"].pop()
    if damage=="mutations":raw["mutated_owned_host_arrays"]=[]
    with pytest.raises(ValueError):probe.validate_raw(raw)

def test_cpu_preflight_cannot_construct_or_create_output(tmp_path, monkeypatch, capsys):
    import sys
    ext=tmp_path/"ropesim/_ropesim.cpython-310-test.so";ext.parent.mkdir();ext.write_bytes(b"CPU")
    output=tmp_path/"output"
    argv=["probe","--extension",str(ext),"--extension-sha256",hashlib.sha256(b"CPU").hexdigest(),
          "--output",str(output)]
    monkeypatch.setattr(probe,"run_probe",lambda *a:pytest.fail("GPU reached"))
    monkeypatch.setattr(sys,"argv",argv);probe.main()
    assert json.loads(capsys.readouterr().out)["native_execution_performed"] is False
    monkeypatch.setattr(sys,"argv",argv+["--execute-native"])
    monkeypatch.delenv(probe.WORKER_ENV,raising=False)
    with pytest.raises(ValueError):probe.main()
    assert not output.exists()

def test_worker_binding_restores_original_constants(monkeypatch,tmp_path):
    original=probe.worker_base
    before=(original.ENTRY,original.WORKER_ENV)
    def verify(*args):
        assert original.ENTRY==probe.ENTRY and original.WORKER_ENV==probe.WORKER_ENV
        raise ValueError("fixture gate")
    monkeypatch.setattr(original,"verify_worker",verify)
    with pytest.raises(ValueError,match="fixture gate"):
        probe.verify_worker(tmp_path,tmp_path/"x","a"*64)
    assert (original.ENTRY,original.WORKER_ENV)==before

@pytest.mark.parametrize("damage",[None,"head","fingerprint","output","gpu"])
def test_real_worker_intent_gate_before_native_probe(tmp_path,monkeypatch,damage):
    import os,shutil,subprocess,sys
    from tests import test_dh116_deformx_pair_worker_v2 as fixture
    monkeypatch.setattr(fixture,"ENTRY",probe.ENTRY)
    monkeypatch.setattr(fixture,"WRAPPER","scripts/run_dh116_deformx_parameter_work_packages.py")
    root,private,path,output,extension,spec,approval=fixture.fixture_repo(tmp_path)
    shutil.copyfile(fixture.ROOT/"scripts/verify_dh116_deformx_pair_capture_v2.py",
                    root/"scripts/verify_dh116_deformx_pair_capture_v2.py")
    intent=json.loads(path.read_text())
    if damage=="head":intent["head"]="a"*40
    if damage=="fingerprint":intent["code_config_sha256"]="f"*64
    if damage=="output":intent["output"]=str(tmp_path/"wrong")
    path.write_text(json.dumps(intent))
    script=("import contextlib,json,sys\\n"
            "from scripts import verify_dh116_deformx_parameter_getters as p\\n"
            "p.native_instance_lock=lambda:contextlib.nullcontext()\\n"
            "p.run_probe=lambda *a:json.loads("+repr(json.dumps(evidence()))+")\\n"
            "p.main()\\n").replace("\\\\n","\\n")
    # Literal source newlines, not a shell interpolation.
    script=script.replace("\\n","\n")
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(root),str(fixture.ROOT/"src"))),
         "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"2" if damage=="gpu" else "3",
         probe.WORKER_ENV:str(path)}
    command=[sys.executable,"-c",script,*[str(output) if x=="{output}" else x for x in spec["argv"]]]
    done=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=20)
    if damage is None:
        assert done.returncode==0,done.stdout+done.stderr
        assert json.loads((output/"result.json").read_text())["snapshot_count"]==3
    else:
        assert done.returncode!=0 and not output.exists(),done.stdout+done.stderr
