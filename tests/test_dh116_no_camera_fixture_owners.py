"""Owner isolation negatives use inert Python sentinels, never native loading."""
from types import SimpleNamespace
import pytest
import torch
from test_dh116_no_camera_collector import fixture

NATIVE_SHAS = (
    "c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53",
    "2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec",
    "e"*64,  # unknown/new backend must not acquire CPU provenance by exclusion
)

@pytest.mark.parametrize("backend", NATIVE_SHAS)
def test_any_non_fixture_backend_is_denied_before_journal(monkeypatch,tmp_path,backend):
    def mutate(ep,rod,hands,sim,rows,r0):
        ep._boundary._sha256=backend
    with pytest.raises(ValueError,match="CPU|fixture|owner|native"):
        fixture(monkeypatch,tmp_path,mutate=mutate)
    assert not (tmp_path/"collector.jsonl").exists()

@pytest.mark.parametrize("slot",["sim","rope","hand","view","exchange"])
def test_native_owner_wrapper_is_denied_even_with_fixture_artifact(monkeypatch,tmp_path,slot):
    native_type=type("NativeOwner",(),{"__module__":"ropesim._ropesim"})
    def mutate(ep,rod,hands,sim,rows,r0):
        target={"sim":sim,"rope":rod.rope,"hand":hands[0],
                "view":hands[0].root_physx_view,"exchange":ep._exchange}[slot]
        target._native=SimpleNamespace(owner=native_type())
    with pytest.raises(ValueError,match="CPU|native|owner"):
        fixture(monkeypatch,tmp_path,mutate=mutate)
    assert not (tmp_path/"collector.jsonl").exists()

@pytest.mark.parametrize("module_name",["isaaclab.sim","omni.physics.tensors","ropesim"])
def test_unknown_owner_class_cannot_copy_fake_fields(monkeypatch,tmp_path,module_name):
    def mutate(ep,rod,hands,sim,rows,r0):
        impersonator=type("SimulationContext",(),{"__module__":module_name})()
        vars(impersonator).update(vars(sim))
        ep._sim=impersonator
        ep._exchange._sim=impersonator
    with pytest.raises(ValueError,match="CPU|native|owner"):
        fixture(monkeypatch,tmp_path,mutate=mutate)
    assert not (tmp_path/"collector.jsonl").exists()

def test_fake_rope_class_name_is_not_a_source_identity(monkeypatch,tmp_path):
    def mutate(ep,rod,hands,sim,rows,r0):
        impostor=type("FakeRope",(),{"__module__":"test_dual_hand_rod_runtime"})()
        vars(impostor).update(vars(rod.rope))
        rod._base.rope=impostor
    with pytest.raises(ValueError,match="CPU|owner"):
        fixture(monkeypatch,tmp_path,mutate=mutate)
    assert not (tmp_path/"collector.jsonl").exists()

def test_cuda_tensor_marker_rejected_without_cuda_allocation(monkeypatch,tmp_path):
    class CUDAMarker(torch.Tensor):
        @property
        def device(self):return torch.device("cuda:0")
    tensor=torch.Tensor._make_subclass(CUDAMarker,torch.zeros(1),False)
    def mutate(ep,rod,hands,sim,rows,r0):
        sim.unexpected=tensor
    with pytest.raises(ValueError,match="CPU|CUDA|owner"):
        fixture(monkeypatch,tmp_path,mutate=mutate)
    assert not (tmp_path/"collector.jsonl").exists()

def test_owner_replacement_after_construction_rejected_before_sdk(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    rod.rope._native=SimpleNamespace(owner=object())
    before=sim.current_time_step_index,rod.rope.steps
    with pytest.raises(ValueError,match="CPU|native|owner"):c.collect()
    assert (sim.current_time_step_index,rod.rope.steps)==before
    c.close()

def test_owner_mutation_between_substeps_stops_before_second_sdk(monkeypatch,tmp_path):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    original=sim.step
    def changed(*,render):
        original(render=render)
        sim._native=SimpleNamespace(owner=object())
    sim.step=changed
    with pytest.raises(ValueError,match="CPU|native|owner"):c.collect()
    assert sim.current_time_step_index==3 and rod.rope.steps==1
    assert not len(tr.rollout)
    c.close()
