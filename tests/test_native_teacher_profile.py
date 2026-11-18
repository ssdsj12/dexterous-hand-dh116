"""C01 CPU source tests; no native import, reset or qualification."""
from dataclasses import FrozenInstanceError
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / "src/go2_pvcnn/control/dh116/tabletop_native_teacher_profile.py"
    assert path.is_file(), "C01 independent BackendProfile implementation missing"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_teacher_profile")


@pytest.fixture(scope="module")
def profile():
    return api().BackendProfile.load(ROOT)


def test_actual_source_profile_has_complete_inventory(profile):
    m = api()
    assert profile.binary_sha256 == "8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340"
    assert profile.source_file_count == 72
    assert profile.basis_file_count == 156
    assert profile.local_file_count == 2
    assert len(profile.inventory) > 158
    assert profile.model_schema == m.MODEL_SCHEMA_VERSION
    assert len(profile.digest) == 64
    profile.verify(ROOT)


def test_profile_is_deeply_immutable(profile):
    with pytest.raises(FrozenInstanceError):
        profile.digest = "0" * 64
    with pytest.raises((AttributeError, TypeError)):
        profile.inventory[0].sha256 = "0" * 64
    with pytest.raises(TypeError):
        profile.inventory[0] = profile.inventory[-1]
    assert hash(profile)


@pytest.mark.parametrize("binary", ["0" * 64, "2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"])
def test_unknown_or_old_backend_cannot_be_relabelled(binary):
    with pytest.raises(ValueError, match="backend"):
        api().BackendProfile.from_sources(binary_sha256=binary, source_inventory={})


def test_caller_inventory_and_ready_bool_cannot_authorize(profile):
    with pytest.raises(TypeError):
        api().BackendProfile.load(ROOT, native_ready=True)
    with pytest.raises(ValueError, match="inventory"):
        api().BackendProfile.from_sources(binary_sha256=profile.binary_sha256, source_inventory={})
    with pytest.raises(TypeError):
        api().BackendProfile()


@pytest.mark.parametrize("kind", ["tamper", "symlink", "directory"])
def test_actual_regular_file_hash_required(tmp_path, kind):
    m = api()
    p = tmp_path / "actual"
    p.write_bytes(b"source")
    sha = hashlib.sha256(b"source").hexdigest()
    if kind == "tamper":
        p.write_bytes(b"other")
    elif kind == "symlink":
        q = tmp_path / "link"
        q.symlink_to(p)
        p = q
    else:
        p = tmp_path
    with pytest.raises(ValueError, match="regular|SHA|symlink"):
        m.checked_file(p, sha)


def test_profile_reopens_bytes_each_verification(profile, monkeypatch):
    m = api()
    original = m.checked_file
    seen = []
    def check(path, sha):
        seen.append(str(path))
        return original(path, sha)
    monkeypatch.setattr(m, "checked_file", check)
    profile.verify(ROOT)
    assert profile.binary_path in seen
    assert len(seen) >= len(profile.inventory)


def test_profile_and_static_imports_do_not_load_native():
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'isaaclab','isaacsim','omni','ropesim','pxr'}:
   raise AssertionError('native import: '+fullname)
sys.meta_path.insert(0, Guard())
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import required_capabilities
p=BackendProfile.load()
assert p.digest
assert required_capabilities('approach_wiring_smoke_v1')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("which", ["binary", "reader", "radius_reader"])
def test_source_loader_rejects_current_source_tamper(monkeypatch, which):
    m = api()
    target = (Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114/"
                   "build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so") if which == "binary"
              else ROOT / ("scripts/replay_dh116_native_parameter_r0.py" if which == "radius_reader"
                           else "src/go2_pvcnn/control/dh116/tabletop_teacher_mounted_wrist_source.py"))
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
        lambda path: original(path) + (b"tamper" if path == target else b""))
    with pytest.raises(ValueError, match="SHA|source"):
        m.BackendProfile.load(ROOT)

def test_pinch_construction_scope_uses_the_observation_backend_then_restores_legacy(profile):
    from pathlib import Path
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope

    legacy = (builder.BACKEND, builder.BACKEND_SHA)
    assert legacy[1] != profile.binary_sha256
    with pytest.raises(RuntimeError, match="synthetic operation"):
        with profile_reader_scope(profile):
            assert (builder.BACKEND, builder.BACKEND_SHA) == (
                Path(profile.binary_path), profile.binary_sha256)
            raise RuntimeError("synthetic operation")
    assert (builder.BACKEND, builder.BACKEND_SHA) == legacy
