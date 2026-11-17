"""Dedicated bounded worker and real parent receipt; no Isaac starts here."""
import importlib.util
import json
from pathlib import Path
import pytest
from test_dh116_pinch_v3 import cpu
from test_dh116_dual_pinch_closure_worker import BLOCKER
ROOT=Path(__file__).resolve().parents[1]
def entry():
    n="scripts.verify_dh116_tripod_owned_contact"
    assert importlib.util.find_spec(n) is not None,"dedicated Tripod worker missing"
    return __import__(n,fromlist=["preflight"])

def test_real_bilateral_parent_and_independent_tripod_identity():
    e=entry();p,inputs=e.load_parent();c=e.preflight(ROOT)
    assert p["qualification_scope"]=="bilateral_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"
    assert inputs[str(e.PARENT_RECEIPT)]==e.PARENT_SHA
    assert c["physics_steps"]==234 and c["parameter_endpoint_count"]==233
    assert c["native_execution_performed"] is False
    assert c["teacher_admission"] is False
    assert set(e.FILES)<=set(c["sources"]["local_sources"])

def test_no_native_import_before_app():
    e=entry()
    r=cpu(BLOCKER+"""
from pathlib import Path
from scripts import verify_dh116_tripod_owned_contact as e
c=e.preflight(Path.cwd())
e.previous.prior.original.prior.assert_clean_import_boundary()
assert c["physics_steps"]==234
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("TRIPOD_CPU_PREFLIGHT")
""")
    assert r.returncode==0,r.stdout+r.stderr
    assert r.stdout.strip()=="TRIPOD_CPU_PREFLIGHT"

def cleanup(spec):
    e=entry()
    from scripts import run_dh116_tripod_owned_work_packages as w
    with e.bindings((w.base,"ENTRY",e.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        return w.cleanup_case(spec)

def test_frozen_source_and_budget():
    e=entry();s=e.read(ROOT/"configs/dh116_execution/deformx_tripod_owned_contact_v1.json")
    c=e.preflight(ROOT)
    assert cleanup(s)=="explicit"
    assert s["entrypoint"]==e.ENTRY
    assert s["validator"]["equals"]["code_config_sha256"]==c["code_config_sha256"]
    assert c["sources"]["external_sources"].items()<=s["inputs"].items()

@pytest.mark.parametrize("key,value",[("physics_steps",240),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_no_budget_expansion(key,value):
    e=entry();s=e.read(ROOT/"configs/dh116_execution/deformx_tripod_owned_contact_v1.json")
    s["budget"][key]=value
    with pytest.raises(ValueError):cleanup(s)

@pytest.mark.parametrize("marker",[False,True])
def test_real_cli_worker_identity_before_app(tmp_path,marker):
    e=entry()
    r=cpu(BLOCKER+"""
import os,json
from pathlib import Path
from scripts import verify_dh116_tripod_owned_contact as e
out=Path(OUTPUT);c=e.preflight(Path.cwd());called=[]
os.environ.pop(e.WORKER_ENV,None)
if ALLOW:
    os.environ[e.WORKER_ENV]=json.dumps(dict(version=e.VERSION,case="explicit",output=str(out.resolve()),
        entry_sha256=e.sha(Path.cwd()/e.ENTRY),wrapper_sha256=e.sha(Path.cwd()/e.WRAPPER)))
os.environ["CUDA_VISIBLE_DEVICES"]="3";os.environ["DH116_PHYSX_LIFETIME_GDB"]="1"
e.execute=lambda *a:called.append(True)
sys.argv=[e.ENTRY,"--cleanup","explicit","--output",str(out),"--headless","--execute-native","--steps","234",
    "--approved-code-config-sha256",c["code_config_sha256"],"--left-scheduler-receipt",str(e.PARENT_RECEIPT),
    "--left-receipt-sha256",e.PARENT_SHA]
try:e.main()
except (ValueError,RuntimeError,SystemExit):
    assert not ALLOW and not called and not out.exists()
else:assert ALLOW and called==[True]
assert not forbidden.intersection({x.split(".")[0] for x in sys.modules})
print("TRIPOD_CLI_CPU")
""".replace("OUTPUT",repr(str(tmp_path/"native"))).replace("ALLOW",repr(marker)))
    assert r.returncode==0,r.stdout+r.stderr
    assert r.stdout.strip()=="TRIPOD_CLI_CPU"
