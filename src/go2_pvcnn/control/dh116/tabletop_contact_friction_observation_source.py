"""Opt-in source for native raw friction readout; old physical defaults stay intact.

This seals observation bytes and the finite standalone probe, not DH116 slip,
Reset, grasp or Teacher qualification. Importing creates no CUDA/Isaac owner.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading

from . import tabletop_mesh_observation_source as parent
from .tabletop_native_teacher_profile import checked_file, digest

VERSION = "dh116-contact-friction-observation-source-v1"
BUILD = parent.BUILD.parent / "contact-friction-observations-20261122"
BACKEND = BUILD / "build/python/ropesim" / parent.BACKEND.name
BACKEND_SHA = "847c1f49c195b275ac23175182194ad0764a235e5adaed9407d8d6dcce251cf4"
TREE_SHA = "0204964dfce03cb95de96f33ddd338519cb925423f992be5774b6166f94e8f91"
PATCH_SHA = "8a02fb50d4d581ffffb8a01b7c94e46fb87f965f8a65def9a2c965981c2c355a"
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/teacher-friction-observation-v1")
DELTA = {
    "csrc/bindings.cpp": "9262dcc7254cf1282d08090073c2fe4351b84559244ee27c1833a9312640aa9b",
    "third_party/yarnball/KittenEngine/YarnBall/YarnBall.h":
        "caa6882abcc8fb2950f3d0194962b24e3ea1f67e328ed8ee270e6781d30b258f",
    "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh":
        "0747070c1bbc797d6d526dc2f808064741bf7f9820fbd28d4510a02cfc6fa1fd",
}
LOCAL_SOURCES = {
    "scripts/prepare_dh116_deformx_contact_friction.py":
        "fb9dfa47b025841e50641657d4ebdcd14397757c4d2652c84918fd0e8e80dd56",
    "scripts/verify_dh116_deformx_contact_friction.py":
        "88fc36721ab1021c4b882e5082bde8b9360c5ca7e24ba2f5e5e1ebdf74e3f8d4",
}


def source_basis():
    """Reopen the full parent/child trees, facades, build and GPU evidence."""
    old = parent.source_basis()
    sources = dict(old["sources"])
    sources.update({
        str(BACKEND): BACKEND_SHA,
        str(BUILD / "build/preparation.json"):
            "390e3088310e704e079923fe6f2612af45f3b889faf2796b749cf043b315c78d",
        str(BUILD / "build/observation-only.patch"): PATCH_SHA,
        str(EVIDENCE / "build-result.json"):
            "d8c078b042e1b9697d36d98571338cd622e614cda6e130bbfda1c110ffcc47fb",
        str(EVIDENCE / "native-probe-v1/outer.json"):
            "2ee0837a5e151bb8f6dee4fee8e1923b5f5fbff9c8dac61753b72665e34c1aad",
        str(EVIDENCE / "native-parity-analysis-v3.json"):
            "4e8b7544512b123c68856337315734b7a649adc4605c62a2ca330103fadcc64b",
        str(EVIDENCE / "native-probe-v1/baseline.json"):
            "97c4833fab11a489807b67ad37923a7134145f7396862d2c780a6d3e7d997fed",
        str(EVIDENCE / "native-probe-v1/candidate.json"):
            "6ddc5c3d0d2f0b7222b47d0eb2c1303c66a9be1ce305f406d06e32a6fafa5c97",
    })
    preparation = json.loads(checked_file(BUILD / "build/preparation.json",
        sources[str(BUILD / "build/preparation.json")]))
    inventory = preparation["prepared_source_sha256"]
    original = json.loads(checked_file(parent.PREPARATION, parent.PREPARATION_SHA))["prepared_source_sha256"]
    if (len(inventory) != 72 or set(inventory) != set(original)
            or digest(inventory) != TREE_SHA
            or {k: v for k, v in inventory.items() if original[k] != v} != DELTA
            or preparation["changed_sources"] != DELTA
            or preparation["delta_sha256"] != PATCH_SHA
            or preparation["original_backend_sha256"] != parent.BACKEND_SHA
            or preparation["original_source_tree_sha256"] != parent.TREE_SHA):
        raise ValueError("complete reviewed friction observation lineage required")
    for name, expected in inventory.items():
        relative = Path(name)
        path = BUILD / "source" / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve(strict=True).is_relative_to(BUILD / "source")):
            raise ValueError("friction source escapes its independent tree")
        sources[str(path)] = expected
    for name in ("__init__.py", "sdf.py"):
        sources[str(BUILD / "build/python/ropesim" / name)] = inventory["ropesim/" + name]
    for path, expected in sources.items():
        checked_file(path, expected)
    for path, expected in LOCAL_SOURCES.items():
        checked_file(parent.ROOT / path, expected)
    return dict(version=VERSION, backend_path=str(BACKEND), backend_sha256=BACKEND_SHA,
        source_tree_sha256=TREE_SHA, source_file_count=72, patch_delta_sha256=PATCH_SHA,
        sources=sources, local_sources={**old["local_sources"], **LOCAL_SOURCES},
        changed_sources=deepcopy(DELTA), parent_binary_sha256=parent.BACKEND_SHA,
        standalone_probe_qualified=True, standalone_scope="three controlled plane-mesh cases",
        native_qualification=False, reset_qualified=False, slip_qualified=False,
        teacher_admission=False)


def verify_runtime(rod):
    """Check the already loaded exact owner without importing another extension."""
    source_basis()
    extension = sys.modules.get("ropesim._ropesim")
    package = sys.modules.get("ropesim")
    if (extension is None or getattr(package, "_ropesim", None) is not extension
            or Path(extension.__file__).resolve() != BACKEND
            or type(rod.rope) is not extension.RopeBatch
            or rod.rope.num_envs != 1 or rod.rope.verts_per_rope != 41
            or not callable(getattr(rod.rope, "contact_friction_samples_v1", None))):
        raise ValueError("exact loaded friction observation owner required")


_lock = threading.RLock()
_depth = 0
_thread = None


@contextmanager
def source_scope():
    """Bind actual parameter/health/build readers, then restore every old default.

    New scopes may nest on one thread. Neither legacy reader/source nesting nor
    a different thread is admitted. This is a synchronous single-owner bridge.
    """
    global _depth, _thread
    if not _lock.acquire(blocking=False):
        raise ValueError("friction source scope belongs to another thread")
    prior = ()
    try:
        from . import tabletop_native_teacher_profile as legacy_profile
        from scripts import replay_dh116_native_parameter_r0 as radius
        owner = threading.get_ident()
        if _depth and _thread != owner:
            raise ValueError("friction source scope belongs to another thread")
        if parent._depth or legacy_profile._profile_reader_depth:
            raise ValueError("friction source requires an idle legacy reader")
        basis = source_basis()
        bindings = (
            (parent._parameters, "VERSION", VERSION),
            (parent._parameters, "BUILD", BUILD),
            (parent._parameters, "BACKEND", BACKEND),
            (parent._parameters, "BACKEND_SHA", BACKEND_SHA),
            (parent._parameters, "source_basis", source_basis),
            (parent._health, "BACKEND", BACKEND),
            (parent._health, "BACKEND_SHA", BACKEND_SHA),
            (parent._builder, "BACKEND", BACKEND),
            (parent._builder, "BACKEND_SHA", BACKEND_SHA),
            (radius, "BACKEND_SHA", BACKEND_SHA),
        )
        expected = bindings if _depth else (*parent._ORIGINAL, (radius, "BACKEND_SHA", parent.OLD_BACKEND_SHA))
        if any(getattr(module, name) != value for module, name, value in expected):
            raise ValueError("unexpected friction source reader binding")
        if not _depth:
            prior = tuple((module, name, getattr(module, name)) for module, name, _ in bindings)
            for module, name, value in bindings:
                setattr(module, name, value)
            _thread = owner
        _depth += 1
        try:
            yield deepcopy(basis)
        finally:
            _depth -= 1
            if not _depth:
                for module, name, value in prior:
                    setattr(module, name, value)
                _thread = None
    finally:
        _lock.release()
