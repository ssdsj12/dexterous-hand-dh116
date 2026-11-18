"""CPU source preparation only: no compilation, import of ropesim, or solver execution."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import pytest

from scripts import prepare_dh116_deformx_mesh_observations as prep

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    "csrc/sdf.h", "csrc/sdf.cu", "csrc/bindings.cpp",
    "third_party/yarnball/KittenEngine/YarnBall/YarnBall.h",
    "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh",
}

@pytest.fixture(scope="module")
def generated():
    return prep.planned_sources(ROOT)

def test_only_five_reviewed_files_change(generated):
    original=prep.locked_sources(ROOT)
    assert len(original)==72
    assert {name for name in original if original[name]!=generated[name]}==EXPECTED

@pytest.mark.parametrize("name",sorted(EXPECTED))
def test_every_delta_reverses_to_exact_locked_source(generated,name):
    text=generated[name]
    for filename,before,after in reversed(prep.PATCHES):
        if filename==name:
            assert text.count(after)==1
            text=text.replace(after,before)
    assert text==prep.locked_sources(ROOT)[name]

def test_old_pair_api_exact_bytes_preserved(generated):
    name="csrc/bindings.cpp"
    def old_api(text):
        return text.split('        .def("contact_pair_samples",',1)[1].split('        .def("contact_forces_view",',1)[0]
    assert old_api(generated[name])==old_api(prep.locked_sources(ROOT)[name])
    assert 'py::array_t<float> values({size, py::ssize_t(12)})' in old_api(generated[name])

def test_pair_payload_uses_same_contact_pose_gap_and_normal_force(generated):
    text=generated["third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh"]
    assert "if (normalForceOut) *normalForceOut = N;" in text
    assert "data->d_sdfPairs ? &nativeNormalForce : nullptr" in text
    assert "P.pos, P.q, gap, nativeNormalForce" in text
    assert text.count("const float N = -k * gap;")==1
    assert text.count("surfaceContact(data, n, gap, dxRel, k, h,")==2

def test_new_pair_api_adds_owned_columns_without_endpoint_estimation(generated):
    text=generated["csrc/bindings.cpp"]
    method=text.split('        .def("contact_pair_samples_v2",',1)[1].split('        .def("mesh_state_snapshot",',1)[0]
    for key in ("schema","indices","vectors","mesh_positions_world_m",
                "mesh_quaternions_xyzw","surface_gap_m","normal_force_n",
                "node_forces_on_rod_world_n","mesh_wrenches_on_mesh_world"):
        assert '["'+key+'"]' in method
    assert "r.meshPosition[j]" in method and "r.meshQuaternion[j]" in method
    assert "r.gap" in method and "r.normalForce" in method
    assert "positions()" not in method and "sdfSample" not in method
    assert "py::keep_alive" not in method

def test_snapshot_synchronizes_and_copies_actual_device_state(generated):
    text=generated["csrc/sdf.cu"].split("SdfStore::MeshState SdfStore::mesh_state_snapshot",1)[1]
    assert text.index("cudaStreamSynchronize(stream)")<text.index("cudaMemcpy(")
    for pointer in ("d_cur","d_tgt","d_tele","d_poses"):
        assert pointer in text
    for forbidden in ("commit(", "begin_step(", "end_step(", "set_targets(", "<<<",
                      "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToDevice"):
        assert forbidden not in text
    assert "result.mesh_count = count()" in text
    assert "result.pair_capacity = pairCapacity" in text

def test_snapshot_binding_owns_arrays_and_marks_source(generated):
    text=generated["csrc/bindings.cpp"].split('        .def("mesh_state_snapshot",',1)[1].split('        .def("contact_pair_samples",',1)[0]
    assert text.index("std::lock_guard<std::mutex>")<text.index("b.sdf.mesh_state_snapshot")
    for key in ("current_positions_world_m","current_quaternions_xyzw",
                "target_positions_world_m","target_quaternions_xyzw",
                "pending_teleport","solver_positions_world_m","solver_quaternions_xyzw",
                "pending_set","moving","pair_capacity","pair_ready"):
        assert '["'+key+'"]' in text
    assert "cuda_device_copy" in text
    assert "py::keep_alive" not in text and "CudaView" not in text

@pytest.mark.parametrize("name",["csrc/bindings.cpp","csrc/sdf.cu","CMakeLists.txt"])
def test_changed_original_source_rejected_before_destination(tmp_path,monkeypatch,name):
    original=prep.sha256
    target=prep.OLD_SOURCE/name
    monkeypatch.setattr(prep,"sha256",lambda path:"0"*64 if Path(path)==target else original(path))
    destination=tmp_path/"new"
    with pytest.raises(ValueError,match="locked source"):
        prep.prepare(ROOT,destination)
    assert not destination.exists()

def test_prepare_is_copy_only_and_commands_are_explicit(tmp_path):
    destination=tmp_path/"new"
    before=prep.sha256(prep.OLD_BACKEND)
    result=prep.prepare(ROOT,destination)
    assert result["compiled"] is False and result["solver_executed"] is False
    assert result["native_qualification"] is False
    assert prep.sha256(prep.OLD_BACKEND)==before
    assert set(result["changed_sources"])==EXPECTED
    for name,expected in result["prepared_source_sha256"].items():
        assert prep.sha256(destination/"source"/name)==expected
    manifest=json.loads((destination/"build/preparation.json").read_text())
    commands=manifest["commands"]
    assert commands["build"][-2:]==["--parallel","2"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in commands["configure"]
    assert commands["environment"]["CUDA_VISIBLE_DEVICES"]==""
    assert commands["time_limits_s"]=={"configure":60,"build":180,"install":30}
    assert manifest["approval_required_before_build"] is True
    assert not list(destination.rglob("*.so"))
    with pytest.raises(ValueError,match="new empty"):
        prep.prepare(ROOT,destination)

def test_protected_source_destination_is_rejected():
    for path in (prep.OLD_SOURCE,prep.OLD_SOURCE/"new",prep.OLD_BACKEND.parent):
        with pytest.raises(ValueError):
            prep.prepare(ROOT,path)

def test_pair_struct_is_additive_and_solver_order_untouched(generated):
    old=prep.locked_sources(ROOT)
    for name in ("csrc/batch.cu","third_party/yarnball/KittenEngine/YarnBall/sim/iteration.cu",
                 "third_party/yarnball/KittenEngine/YarnBall/sim/cosserat.cu"):
        assert generated[name]==old[name]
    header=generated["third_party/yarnball/KittenEngine/YarnBall/YarnBall.h"]
    assert "vec3 point, normal, forceOnRod, torqueOnMesh;" in header
    assert "vec3 meshPosition;" in header and "vec4 meshQuaternion;" in header

def test_native_timestamps_are_readback_owner_values_only(generated):
    text=generated["csrc/bindings.cpp"]
    new=text.split('        .def("contact_pair_samples_v2",',1)[1].split('        .def("contact_pair_samples",',1)[0]
    assert new.count('["owner_time_s_before"]')==2
    assert new.count('["owner_time_s_after"]')==2
    assert new.count("const float before = b.raw()->meta.time;")==2
    assert '["substep_timestamp_s"] = py::none()' in new
    assert "/480" not in new and "(sub + 1)" not in new and "step_free" not in new
