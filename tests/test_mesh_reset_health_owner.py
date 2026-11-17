"""New observer/real Python owner on CPU only; all SDK/solver transport is explicit doubles."""
from copy import deepcopy
import importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_flexible_parameter_readback import case
from test_r0_numerical_health_source import row_for
from test_dh116_dual_pinch_builder import prepared,geometry
ROOT=Path(__file__).resolve().parents[1]

def api():
    assert (ROOT/"src/go2_pvcnn/control/dh116/tabletop_mesh_observation_evidence.py").exists()
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_observation_evidence")
def operation():
    assert (ROOT/"scripts/prepare_dh116_mesh_reset_health_projected_pinch.py").exists()
    return importlib.import_module("scripts.prepare_dh116_mesh_reset_health_projected_pinch")

def empty_raw(time):
    return dict(schema="dh116.sdf_pair_samples.v2",substeps=12,mesh_count=11,capacity=5412,attempted=0,
        overflow=False,indices=np.empty((0,3),np.int32),vectors=np.empty((0,12),np.float32),
        node_forces_on_rod_world_n=np.zeros((41,3),np.float32),mesh_wrenches_on_mesh_world=np.zeros((11,6),np.float32),
        mesh_positions_world_m=np.empty((0,3),np.float32),mesh_quaternions_xyzw=np.empty((0,4),np.float32),
        surface_gap_m=np.empty((0,),np.float32),normal_force_n=np.empty((0,),np.float32),
        sample_source="cuda_surfaceReadout_same_end_substep_evaluation",
        normal_force_semantics="unaveraged_penalty_normal_force_before_friction",
        owner_time_s_before=time,owner_time_s_after=time,owner_time_source="native_host_owner_metadata_at_readback",
        substep_timestamp_s=None,last_error=0)

def prepare_case(c,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    c.base.physics_step=3;c.base.rod_tick=12;c.rope.time=1/480
    c.sim.current_time_step_index=3;c.sim.current_time=3/480;c.expected=c.runtime.clock()
    c.backend.__file__=str(source.BACKEND)
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",c.backend)
    monkeypatch.setitem(sys.modules,"ropesim",SimpleNamespace(_ropesim=c.backend))
    c.rope.contact_pair_samples_v2=lambda:empty_raw(c.rope.time)
    return source

def test_one_light_getter_keeps_actual_clock_and_has_no_source_file_or_parser_work(case,monkeypatch):
    c=case;source=prepare_case(c,monkeypatch);m=api();calls=[]
    with source.source_scope():
        row=row_for(c)
        c.rope.contact_pair_samples_v2=lambda:calls.append("v2") or empty_raw(c.rope.time)
        def forbidden(*a,**k):raise AssertionError("heavy source/parser called online")
        monkeypatch.setattr(source,"source_basis",forbidden)
        monkeypatch.setattr(m.pairs,"parse_pair_observations",forbidden)
        got=m.capture_pair_v2(c.runtime,sim=c.sim,exchange=row)
        assert m.validate_pair_boundary(got,row)["source_complete"]
        assert calls==["v2"] and c.events==[]
        assert got["boundary_before"]==got["boundary_after"]
        assert got["new_sdk_steps"]==got["new_solver_steps"]==0

@pytest.mark.parametrize("failure",["getter","sdk","solver","owner","schema","count"])
@pytest.mark.parametrize("old_guard",[None,"post_step_source_clearance:right"])
def test_completed_row_survives_getter_fault_and_keeps_original_guard(case,monkeypatch,failure,old_guard):
    c=case;source=prepare_case(c,monkeypatch);m=api()
    def bad():
        raw=empty_raw(c.rope.time)
        if failure=="getter":raise RuntimeError("v2 fixture")
        if failure=="sdk":c.sim.current_time+=.001
        if failure=="solver":c.rope.time+=.001
        if failure=="owner":c.backend.RopeBatch=object
        if failure=="schema":raw["schema"]="wrong"
        if failure=="count":raw["attempted"]=1
        return raw
    c.rope.contact_pair_samples_v2=bad
    with source.source_scope():
        row=row_for(c);row["guard_failure"]=old_guard
        class Parent:
            def _step(self,action,substep):return row
        class Child(m.PairV2EvidenceMixin,Parent):pass
        owner=Child();owner._rod=c.runtime;owner._sim=c.sim
        got=owner._step(None,0)
        assert got is row and got["mesh_pair_v2"]["capture_error"]
        assert got["guard_failure"]==(old_guard or "mesh_pair_v2_source_invalid")
        assert got["mesh_observation_prior_guard_failure"]==old_guard

def test_scope_is_nested_once_and_restores_original_versions():
    op=operation()
    from scripts import prepare_dh116_health_projected_pinch as health
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    before=(health.VERSION,health.SCOPE,health.scope_marker,director.DirectorExchange,parameters.VERSION)
    with op.scenario_scope():
        kind=director.DirectorExchange
        assert sum(c is api().PairV2EvidenceMixin for c in kind.__mro__)==1
        assert health.scope_marker()==op.scope_marker()
        with op.scenario_scope():assert director.DirectorExchange is kind
        assert parameters.BACKEND_SHA.startswith("881701")
    assert before==(health.VERSION,health.SCOPE,health.scope_marker,director.DirectorExchange,parameters.VERSION)


def test_complete_capture_is_bound_by_production_scope_and_restores_after_exception():
    op=operation()
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
    from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
    original=runtime.read_pair_capture
    with pytest.raises(RuntimeError,match="CPU scope exit"):
        with op.scenario_scope():
            assert runtime.read_pair_capture is read_complete_pair_capture
            with op.scenario_scope():
                assert runtime.read_pair_capture is read_complete_pair_capture
            assert runtime.read_pair_capture is read_complete_pair_capture
            raise RuntimeError("CPU scope exit")
    assert runtime.read_pair_capture is original


def install_mesh_backend(monkeypatch,old,stats,*,failure=None):
    """Extend only the CPU solver transport BEFORE builder's first real Reset."""
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order import _cpu_sdf
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    setup=old.prior_test.setup
    def enriched(*a,**k):
        values=setup(*a,**k);sim=values[0]
        package=sys.modules["ropesim"];backend=package._ropesim;Rope=package.RopeBatch
        backend.RopeBatch=Rope
        monkeypatch.setitem(sys.modules,"ropesim._ropesim",backend)
        sdf=_cpu_sdf(source.BUILD/"build/python/ropesim/sdf.py")
        package.sdf=sdf;monkeypatch.setitem(sys.modules,"ropesim.sdf",sdf)
        old_create=Rope.from_positions;old_add=Rope.add_mesh
        old_step=Rope.step_free;old_targets=Rope.set_mesh_poses;old_enable=Rope.enable_contact_pair_capture
        cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())["rod"]
        def create(cls,nodes,**kw):
            obj=old_create(nodes,**kw);obj.verts_per_rope=41;obj.last_error=0
            obj.cur={};obj.tgt={};obj.tele={};obj.capacity=0;obj.pending=False;obj.ready=False;obj.clears=0
            return obj
        def add(self,*a,**k):
            i=old_add(self,*a,**k);q=np.asarray(k.get("quat",[0,0,0,1]),np.float32).reshape(-1,4)[0].copy()
            q/=np.linalg.norm(q);pos=np.asarray(k["pos"],np.float32).reshape(-1,3)[0].copy()
            self.cur[i]=(pos,q);self.tgt[i]=(pos.copy(),q.copy());self.tele[i]=0;self.ready=False
            return i
        def clear(self):
            self.slots.clear();self.cur.clear();self.tgt.clear();self.tele.clear()
            self.wrenches=np.zeros((1,0,6));self.pending=False;self.ready=False;self.clears+=1
        def target(self,i,p,q,*,teleport):
            old_targets(self,i,p,q,teleport=teleport)
            quat=np.asarray(q,np.float32).reshape(-1,4)[0].copy();quat/=np.linalg.norm(quat)
            self.tgt[i]=(np.asarray(p,np.float32).reshape(-1,3)[0].copy(),quat)
            self.tele[i]=int(teleport);self.pending=True
        def step(self,dt,substeps):
            old_step(self,dt,substeps)
            self.cur=deepcopy(self.tgt);self.tele={i:0 for i in self.cur};self.pending=False;self.ready=True
        def enable(self,capacity):
            old_enable(self,capacity);self.capacity=capacity
        def quaternion(self):
            q=np.zeros((1,41,4),np.float32);q[...,3]=1;q[0,-1]=[2,3,4,5]
            return q
        def parameters(self):
            rest=np.linalg.norm(np.diff(self.nodes[0],axis=0),axis=1)[None]
            return dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=self.time,
                metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
                contact_radius_m=cfg["radius_m"],gravity_world_m_s2=np.asarray(cfg["gravity_world_m_s2"],np.float32),
                contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],contact_v_slip_m_s=cfg["v_slip_m_s"],
                contact_kappa=cfg["kappa"],rest_lengths_m=rest,stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
                rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
                stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"],np.float32),
                bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"],np.float32),
                inverse_masses_kg_inv=self.inv_masses()[None])
        def mesh(self):
            stats.setdefault("mesh_sdk",[]).append(sim.current_time_step_index)
            if failure=="reset" and self.clears:raise ValueError("CPU reset mesh getter failure")
            pos=np.array([[self.cur[i][0] for i in range(11)]],np.float32)
            q=np.array([[self.cur[i][1] for i in range(11)]],np.float32)
            return dict(schema="dh116.mesh_state_snapshot.v1",num_envs=1,mesh_count=11,
                current_positions_world_m=pos,current_quaternions_xyzw=q,
                target_positions_world_m=np.array([[self.tgt[i][0] for i in range(11)]],np.float32),
                target_quaternions_xyzw=np.array([[self.tgt[i][1] for i in range(11)]],np.float32),
                pending_teleport=np.array([[self.tele[i] for i in range(11)]],np.uint8),
                solver_positions_world_m=pos.copy(),solver_quaternions_xyzw=q.copy(),
                device_state_source="cuda_device_copy",pending_set=self.pending,moving=False,
                pair_capacity=self.capacity,pair_ready=self.ready,flags_source="native_host_sdf_store",
                owner_time_s_before=self.time,owner_time_s_after=self.time,
                owner_time_source="native_host_owner_metadata_at_readback",last_error=0)
        for name,method in dict(from_positions=classmethod(create),add_mesh=add,clear_meshes=clear,
            set_mesh_poses=target,step_free=step,enable_contact_pair_capture=enable,
            quaternions=quaternion,parameter_snapshot=parameters,mesh_state_snapshot=mesh).items():
            monkeypatch.setattr(Rope,name,method,raising=False)
        return values
    monkeypatch.setattr(old.prior_test,"setup",enriched)
    prior_install=old.prior_test.install_sdk_execution
    def install(sim,rod,hand_bindings,events):
        prior_install(sim,rod,hand_bindings,events)
        rope=rod.rope
        def v1():
            ids=np.array([[i,0,j] for i in range(12) for j in (0,6)],np.int32)
            vectors=np.zeros((24,12),np.float32);vectors[:,5]=1;vectors[:,8]=np.tile([1.,2.],12)
            vectors[:,:3]=np.array([rope.cur[int(j)][0] for _,_,j in ids],np.float32)
            return dict(substeps=12,mesh_count=11,capacity=5412,attempted=24,overflow=False,
                indices=ids,vectors=vectors,node_forces_on_rod_world_n=rope.force[0].astype(np.float32),
                mesh_wrenches_on_mesh_world=rope.wrenches[0].astype(np.float32))
        def v2():
            stats.setdefault("v2_sdk",[]).append(sim.current_time_step_index)
            if failure=="pair" and sim.current_time_step_index==11:raise ValueError("CPU v2 getter failure")
            raw=v1();indices=raw["indices"]
            return dict(raw,schema="dh116.sdf_pair_samples.v2",
                mesh_positions_world_m=np.array([rope.cur[int(j)][0] for _,_,j in indices],np.float32),
                mesh_quaternions_xyzw=np.array([rope.cur[int(j)][1] for _,_,j in indices],np.float32),
                surface_gap_m=np.full(24,-.0001,np.float32),normal_force_n=np.tile([1.,2.],12).astype(np.float32),
                sample_source="cuda_surfaceReadout_same_end_substep_evaluation",
                normal_force_semantics="unaveraged_penalty_normal_force_before_friction",
                owner_time_s_before=rope.time,owner_time_s_after=rope.time,
                owner_time_source="native_host_owner_metadata_at_readback",substep_timestamp_s=None,last_error=0)
        rope.contact_pair_samples=v1;rope.contact_pair_samples_v2=v2
    monkeypatch.setattr(old.prior_test,"install_sdk_execution",install)

@pytest.mark.parametrize("failure,old_guard",[(None,False),("pair",False),("pair",True),("reset",False)])
def test_real_full_owner_or_failed_actual_prefix(monkeypatch,tmp_path,prepared,geometry,failure,old_guard):
    import test_left_pinch_approach_owner as old
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    op=operation();stats={}
    install_mesh_backend(monkeypatch,old,stats,failure=failure)
    # Exercise the production scenario scope; no test-only capture binding.
    with op.scenario_scope(),bindings((old,"closure",op)):
        if failure and not old_guard:
            with pytest.raises(RuntimeError,match="mesh"):
                old.run_owner(monkeypatch,tmp_path,prepared,geometry)
        else:old.run_owner(monkeypatch,tmp_path,prepared,geometry,old_guard=old_guard)
    out=tmp_path/"closure"
    count=0 if failure=="reset" else 9 if failure else 232
    rows=[json.loads(line) for line in (out/"physical.jsonl").read_text().splitlines()]
    physical=[decode_owned(r["payload"])["exchange"] for r in rows if r.get("operation")=="physical"]
    assert len(physical)==count
    assert stats["mesh_sdk"]==[2,2]
    assert stats.get("v2_sdk",[])==list(range(3,3+count))
    assert (out/"app_close_returned.json").exists()
    result=op.inspect_observations(out,complete=failure is None)
    assert result["journal"]["measured_physics_rows"]==count
    assert result["reset_mesh_qualified"] is result["substep_contact_frame_qualified"] is False
    if failure=="pair":
        assert physical[-1]["guard_failure"]==("post_step_source_clearance:right" if old_guard else api().REASON)
        assert physical[-1]["mesh_pair_v2"]["capture_error"]
        assert result["pair_v2_audit"]["valid_packets"]==8
        assert result["pair_v2_audit"]["capture_error_rows"]==1
        assert sum(r.get("operation")=="end" for r in rows)==2
    elif failure=="reset":
        state=decode_owned(json.loads((out/api().RESET_FILE).read_text()))
        assert state["readback"]["complete"] is False
        assert state["readback"]["completed_reset_receipt"]["status"]=="complete"
    else:
        assert result["pair_v2_audit"]["valid_packets"]==232
        # Only this explicit synthetic parent gate exercises the pure qualifier;
        # this test is not a real native receipt.
        full=op.inspect_observations(out,parent_lifecycle_qualified=True)
        assert full["reset_mesh_qualified"] and full["substep_contact_frame_qualified"]
        assert full["teacher_admission"] is full["contact_success_qualified"] is False
        assert result["health_getter_audit"]["source_complete"]
