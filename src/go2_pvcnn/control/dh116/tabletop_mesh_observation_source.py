"""Explicit observation backend profile; old defaults are restored outside its scope.

This attests source/build bytes, not a successful native getter or physical reset.
The scope is single-thread owned and must wrap live execution AND post-exit audit.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading

VERSION="dh116-mesh-observation-source-v1"
PARAMETER_VERSION="dh116-mesh-observation-parameter-endpoints-v1"
ROOT=Path(__file__).resolve().parents[4]
BUILD=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114")
BACKEND=BUILD/"build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
BACKEND_SHA="8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340"
OLD_BUILD=BUILD.parent/"parameter-getters-build-e96wtu"
OLD_BACKEND_SHA="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
TREE_SHA="34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64"
OLD_TREE_SHA="3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472"
PREPARATION=BUILD/"build/preparation.json"
PREPARATION_SHA="612696a63262571ab9910ea88421f532f583e7bdb4174851f21a46de662537b9"
PATCH=BUILD/"build/observation-only.patch"
PATCH_SHA="1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41"
PATCHER="scripts/prepare_dh116_deformx_mesh_observations.py"
PATCHER_SHA="213115e0ea79195769a386d94306e1e30dac7a08c8a2317199e67bf927939d45"
LOCK="docs/contracts/deformx_parameter_getter_source_lock_v1.json"
LOCK_SHA="6134aff73fe7692f4a51d9fa18347cfd093648c1ad2854891850900b26a018fe"
CONFIG="configs/dh116_tasks/flexible_pinch_candidate_v1.json"
CONFIG_SHA="9f751680de805c5075282c16bc5c5e5e30b8b7c261a3dd36f3b7b1cbab8f35c8"
EVIDENCE=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-observation-prep-cpu-cca16or5")
BUILD_RECEIPT=EVIDENCE/"isolated-build/result.json"
BUILD_RECEIPT_SHA="e04d1f55493bbb1dd21309cc1e25232980866146ca42fe9811e57e7171fd4038"
IMPORT_RECEIPT=EVIDENCE/"isolated-import/result.json"
IMPORT_RECEIPT_SHA="7fcee5aecaa1fdd9d6faa42f3c88cffbe9e5520ef8df9568110d03f266ce6bdd"
DELTA={
    "csrc/bindings.cpp":"a3c87883962663358c90ccbacdb85596da965d979e8d12baef9c32bf90e30e09",
    "csrc/sdf.cu":"7060092f8782a00d9a692b3639548c76b22447a72006043981f760b40299ae59",
    "csrc/sdf.h":"ab6e7efbedb0a3ca54b2bac9a5be2809b4fc3af6e001810d60c4dea9853661cc",
    "third_party/yarnball/KittenEngine/YarnBall/YarnBall.h":"0291d4565c673fe790e4e4b1bddc8b256eb8651b6cde64ece26b33a67a09035d",
    "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh":"ea3652bcaf05571986fc40e12717ca91ab9debe578eb61f7df4b0bfe35d8201f"}

def _require(ok,message):
    if not ok:raise ValueError(message)

def _read(path,expected):
    path=Path(path)
    _require(not path.is_symlink(),"source symlink is not permitted: "+str(path))
    data=path.read_bytes()
    _require(hashlib.sha256(data).hexdigest()==expected,"locked observation source differs: "+str(path))
    return data

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def source_basis():
    """Validate all 72 prepared files, exactly five deltas, binary and durable receipts."""
    lock=json.loads(_read(ROOT/LOCK,LOCK_SHA));old=lock["files"]
    preparation=json.loads(_read(PREPARATION,PREPARATION_SHA));new=preparation["prepared_source_sha256"]
    _require(len(new)==len(old)==72 and set(new)==set(old) and _digest(new)==TREE_SHA
        and _digest(old)==OLD_TREE_SHA,"complete source inventory differs")
    _require({k:v for k,v in new.items() if old[k]!=v}==DELTA==preparation["changed_sources"],
        "exact five reviewed observation deltas required")
    sources={str(BACKEND):BACKEND_SHA,str(PREPARATION):PREPARATION_SHA,str(PATCH):PATCH_SHA,
        str(BUILD_RECEIPT):BUILD_RECEIPT_SHA,str(IMPORT_RECEIPT):IMPORT_RECEIPT_SHA,
        str(OLD_BUILD/"build/python/ropesim"/BACKEND.name):OLD_BACKEND_SHA,str(ROOT/CONFIG):CONFIG_SHA,
        str(EVIDENCE/"isolated-import/command.json"):"5e932b7447b2ac0360fa0752d78e9f527cf230408c9afea75cc3736fb247ab9b",
        str(EVIDENCE/"isolated-import/stdout.txt"):"1ca0d592d78f22251b256398d36ffc89bbeef111b310cab5809164cc99381d27",
        str(EVIDENCE/"isolated-import/stderr.txt"):"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
    for inventory,build in ((old,OLD_BUILD),(new,BUILD)):
        for name,expected in inventory.items():
            path=build/"source"/name
            _require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.resolve(strict=True).is_relative_to(build/"source"),"source escapes exact build tree")
            sources[str(path)]=expected
    for name in ("__init__.py","sdf.py"):
        sources[str(BUILD/"build/python/ropesim"/name)]=new["ropesim/"+name]
    for path,expected in sources.items():_read(path,expected)
    _read(ROOT/PATCHER,PATCHER_SHA)
    built=json.loads(_read(BUILD_RECEIPT,BUILD_RECEIPT_SHA))
    imported=json.loads(_read(IMPORT_RECEIPT,IMPORT_RECEIPT_SHA))
    _require(built["build_passed"] is True and built["extension_path"]==str(BACKEND)
        and built["extension_sha256"]==BACKEND_SHA and built["source_unchanged"] is True
        and built["old_backend_unchanged"] is True and built["solver_executed"] is False
        and built["api_qualified"] is False and built["gpu_qualified"] is False
        and built["reset_qualified"] is False and built["teacher_qualified"] is False,
        "build receipt is not the independent observation-only binary")
    _require(imported["exit_code"]==0 and imported["timed_out"] is False,"CPU import receipt differs")
    _require(preparation["delta_sha256"]==PATCH_SHA
        and preparation["original_backend_sha256"]==OLD_BACKEND_SHA
        and preparation["original_source_tree_sha256"]==OLD_TREE_SHA,"source lineage differs")
    return dict(version=VERSION,backend_path=str(BACKEND),backend_sha256=BACKEND_SHA,
        source_tree_sha256=TREE_SHA,source_file_count=72,changed_sources=deepcopy(DELTA),
        patch_delta_sha256=PATCH_SHA,sources=sources,
        local_sources={PATCHER:PATCHER_SHA,LOCK:LOCK_SHA},
        only_observation_delta_added=True,compiled=True,cpu_import_only=True,
        standalone_probe_qualified=False,native_qualification=False,task_physics_qualified=False,
        reset_qualified=False,contact_qualified=False,teacher_admission=False,hardware_measured=False)

def verify_runtime(rod):
    """Verify the already loaded class/instance; never import another extension."""
    backend=package=None
    try:
        source_basis()
        backend=sys.modules.get("ropesim._ropesim");package=sys.modules.get("ropesim")
        _require(backend is not None and getattr(package,"_ropesim",None) is backend
            and Path(backend.__file__).resolve()==BACKEND
            and type(rod.rope) is backend.RopeBatch and rod.rope.num_envs==1
            and rod.rope.verts_per_rope==41,"exact loaded observation backend owner required")
    finally:rod=backend=package=None

# These are Python modules only; no ropesim or Isaac import/instance is made.
from . import tabletop_native_parameter_endpoints as _parameters
from . import tabletop_r0_health_separation as _health
from . import tabletop_dual_pinch_builder as _builder
_BINDINGS=((_parameters,"VERSION",PARAMETER_VERSION),(_parameters,"BUILD",BUILD),
    (_parameters,"BACKEND",BACKEND),(_parameters,"BACKEND_SHA",BACKEND_SHA),
    (_parameters,"source_basis",source_basis),(_health,"BACKEND",BACKEND),
    (_health,"BACKEND_SHA",BACKEND_SHA),(_builder,"BACKEND",BACKEND),(_builder,"BACKEND_SHA",BACKEND_SHA))
_ORIGINAL=tuple((module,name,getattr(module,name)) for module,name,_ in _BINDINGS)
_depth=0
_thread=None

@contextmanager
def source_scope():
    """Single-owner opt-in binding, including from-import aliases used by health.

    Same-profile nesting is allowed for existing execute/run/audit scopes; another
    thread or any foreign partial binding fails before mutation. Every prior
    default is restored, including when the caller raises.
    """
    global _depth,_thread
    basis=source_basis();owner=threading.get_ident()
    expected=_BINDINGS if _depth else _ORIGINAL
    _require(_depth==0 or _thread==owner,"source scope belongs to another thread")
    _require(all(getattr(module,name)==value for module,name,value in expected),
        "unexpected observation source binding")
    if _depth==0:
        for module,name,value in _BINDINGS:setattr(module,name,value)
        _thread=owner
    _depth+=1
    try:yield deepcopy(basis)
    finally:
        _depth-=1
        if _depth==0:
            for module,name,value in _ORIGINAL:setattr(module,name,value)
            _thread=None
