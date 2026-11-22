"""Independent source profile and restoration at the actual reader boundary."""
from copy import copy
import importlib
import threading

import pytest


def module():
    name = "go2_pvcnn.control.dh116.tabletop_native_friction_profile"
    assert importlib.util.find_spec(name) is not None, "friction backend profile is missing"
    return importlib.import_module(name)


def test_real_complete_source_and_profile_are_distinct():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as old
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    source = module().source
    basis = source.source_basis()
    assert basis["source_file_count"] == 72
    assert basis["backend_sha256"] != old.BACKEND_SHA
    assert basis["standalone_probe_qualified"] is True
    assert basis["native_qualification"] is False
    assert basis["slip_qualified"] is False
    assert type(profile) is not BackendProfile
    assert profile.parent.binary_sha256 == old.BACKEND_SHA
    assert profile.binary_sha256 == basis["backend_sha256"]
    assert len(profile.inventory) > len(profile.parent.inventory)
    assert profile.verify() is profile


def test_scope_routes_readers_and_restores_after_exception():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    from scripts import replay_dh116_native_parameter_r0 as radius
    m = module()
    before = (parameters.BACKEND_SHA, parameters.source_basis, radius.BACKEND_SHA)
    with pytest.raises(RuntimeError, match="caller failure"):
        with m.friction_profile_reader_scope(profile) as basis:
            assert parameters.BACKEND_SHA == radius.BACKEND_SHA == profile.binary_sha256
            assert parameters.source_basis()["backend_sha256"] == profile.binary_sha256
            assert basis["backend_sha256"] == profile.binary_sha256
            raise RuntimeError("caller failure")
    assert (parameters.BACKEND_SHA, parameters.source_basis, radius.BACKEND_SHA) == before


def test_legacy_scope_and_new_scope_cannot_nest():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as old
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
    m = module()
    with m.friction_profile_reader_scope(profile):
        with pytest.raises(ValueError, match="binding"):
            with old.source_scope():
                pytest.fail("old source entered the friction scope")
        with pytest.raises(ValueError, match="BackendProfile"):
            with profile_reader_scope(profile):
                pytest.fail("old exact-type gate accepted new profile")
    with old.source_scope():
        with pytest.raises(ValueError, match="idle|legacy"):
            with m.friction_profile_reader_scope(profile):
                pytest.fail("friction source entered the legacy scope")


def test_profile_tampering_fails_before_reader_mutation():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    prior = parameters.BACKEND_SHA
    forged = copy(profile)
    object.__setattr__(forged, "binary_sha256", "1"*64)
    with pytest.raises(ValueError, match="identity"):
        with module().friction_profile_reader_scope(forged):
            pytest.fail("tampered profile accepted")
    assert parameters.BACKEND_SHA == prior


def test_other_thread_is_rejected_and_does_not_change_live_reader():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    m = module()
    failures = []
    with m.friction_profile_reader_scope(profile):
        def other():
            try:
                with m.friction_profile_reader_scope(profile):
                    failures.append("unexpected entry")
            except ValueError as error:
                failures.append(str(error))
        worker = threading.Thread(target=other)
        worker.start()
        worker.join(timeout=20)
        assert not worker.is_alive()
        assert len(failures) == 1 and "thread" in failures[0]
        assert parameters.BACKEND_SHA == profile.binary_sha256


def test_same_thread_nested_scope_keeps_outer_binding():
    profile = module().FrictionBackendProfileV2.load()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    m = module()
    prior = parameters.BACKEND_SHA
    with m.friction_profile_reader_scope(profile):
        with m.friction_profile_reader_scope(profile):
            assert parameters.BACKEND_SHA == profile.binary_sha256
        assert parameters.BACKEND_SHA == profile.binary_sha256
    assert parameters.BACKEND_SHA == prior

def test_nested_profile_cannot_swap_self_consistent_identity():
    from dataclasses import asdict
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
    profile = module().FrictionBackendProfileV2.load()
    forged = copy(profile)
    object.__setattr__(forged, "version", "invented-version")
    object.__setattr__(forged, "digest", digest({k:v for k,v in asdict(forged).items() if k != "digest"}))
    with module().friction_profile_reader_scope(profile):
        with pytest.raises(ValueError, match="identity|owner"):
            with module().friction_profile_reader_scope(forged):
                pytest.fail("nested source accepted caller invented profile")
