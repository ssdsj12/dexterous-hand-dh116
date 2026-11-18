"""Observe original frozen native physics; no action or reward modification."""
from copy import deepcopy
import json
import os
import weakref
from pathlib import Path
from scripts import prepare_dh116_effective_right_native as previous
from scripts import prepare_dh116_native_parameter_task as parameter
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_deformable_dual_evidence as evidence
from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as reset_owner
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned,_copy,_digest
from go2_pvcnn.control.dh116.tabletop_r0_live_observer import LiveR0Observer
from go2_pvcnn.control.dh116.tabletop_r0_after_physical_reset import reset_clock_matches
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS
from scripts import replay_dh116_native_parameter_r0 as radius_contract

VERSION="dh116-live-r0-reset-native-v1"
SCOPE="actual_coupled_reset_then_live_original_R0_observer_234_steps_only"
CONFIG="configs/dh116_training/live_r0_reset_native_v1.json"
ROOT=Path(__file__).resolve().parents[1]
original_run=previous.run_existing_app
Journal=evidence.DualHandEvidenceJournal

def write(path,packet):
    with Path(path).open("x") as f:
        json.dump(encode_owned(packet),f,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())

class LiveSession:
    def __init__(self,project,output):
        self.project=Path(project);self.output=Path(output)
        cfg=json.loads((self.project/CONFIG).read_bytes())
        if cfg!=dict(version=VERSION,enabled_by_default=False,physics_steps=234,ppo_updates=0,
            diagnostic_coefficients="existing_ordinal_fixture_not_training",
            training_reward_enabled=False,legacy_physics_actions_unchanged=True):
            raise ValueError("exact isolated live observer contract required")
        self.native_refs={}
        self.reset_event=None
        self.observer=None
        self.stream=None
        self.physical=0;self.controls=0
        self.initialized=False
        self.failure=None

    def emit(self,op,payload):
        if self.stream is None:self.stream=(self.output/"live_r0.jsonl").open("x")
        self.stream.write(json.dumps(encode_owned(dict(operation=op,payload=payload)),allow_nan=False)+"\n")
        self.stream.flush()
        if op!="physical":os.fsync(self.stream.fileno())

    def initialize(self,packet,error,identity,write_original):
        sim=rod=None
        try:
            if self.reset_event is None or set(self.native_refs)!={"sim","rod"}:
                raise ValueError("actual completed transaction and live owners required before R0")
            if error is not None or not isinstance(packet,dict):
                raise ValueError("actual native parameter capture required")
            sim,rod=self.native_refs["sim"](),self.native_refs["rod"]()
            if sim is None or rod is None:raise ValueError("live Reset owner expired before initialization")
            before=reset_owner._clock(sim,rod)
            reset_clock_matches(identity,before["rod"])
            reset_clock_matches(self.reset_event["reset"]["after_clock"],before["rod"])
            radius=radius_contract.check_radius_packet(packet,self.reset_event["reset"]["rod_reset"])
            task=load_flexible_grasp_task(self.project/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",self.project)
            budget=json.loads((self.project/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
            budget["setup_physics_steps_per_env"]=2
            budget["physics"]["solver_substeps_per_physics_step"]=12
            self.observer=LiveR0Observer(task,"pinch",env_id=0,budget_config=budget,
                diagnostic_coefficients={k:float(i+1) for i,k in enumerate(R0_TERMS)},
                backend_artifact=radius_contract.BACKEND,backend_sha256=radius_contract.BACKEND_SHA,
                radius_sample=dict(value=radius["value"],source="native_parameter_snapshot:"+_digest(radius)))
            init=self.observer.complete_physical_reset(self.reset_event,reset_approach=None)
            after=reset_owner._clock(sim,rod)
            if before!=after:raise ValueError("SDK or solver advanced during R0 initialization")
            self.native_refs.clear()
            receipt=dict(version=VERSION,initialization=init,live_clock_before=before,live_clock_after=after,
                native_parameter_sha256=_digest(packet),temporary_native_references_released=True,
                native_qualification=False,teacher_admission=False,legacy_six_hook_order_claimed=False)
            write(self.output/"live_r0_reset.json",receipt)
            self.initialized=True
            self.emit("reset",receipt)
        finally:
            self.native_refs.clear()
            sim=rod=None

    def record(self,row):
        if row["kind"]=="reset":
            if self.reset_event is not None:raise ValueError("one live Reset per bounded recipe")
            self.reset_event=_copy(row)
        elif row["kind"]=="physical":
            if not self.initialized:raise RuntimeError("live R0 initialization required before first physics")
            self.observer.record(row)
            self.physical+=1
            # Immediate owned per-substep record, including a terminal sample.
            self.emit("physical",dict(identity=_copy(row["exchange"]["action_readback"]["post"]["identity"][0]),
                actual_event_sha256=_digest(row),r0_row=self.observer._rows[-1]))
            if row["exchange"]["guard_failure"] is not None:os.fsync(self.stream.fileno())

    def begin(self,pre,action):
        if not self.initialized:raise RuntimeError("live Reset must precede first decision")
        self.observer.begin(pre,action)
        self.emit("begin",dict(pre_action_sha256=_digest(pre),action_sha256=_digest(action)))

    def end(self,row):
        receipt=self.observer.end(row)
        self.controls+=1
        self.emit("end",receipt)

    def close(self):
        self.native_refs.clear()
        if self.stream is not None:
            try:self.stream.flush();os.fsync(self.stream.fileno())
            finally:self.stream.close();self.stream=None
        complete=self.initialized and self.physical==232 and self.controls==29 and self.failure is None
        write(self.output/"live_r0_summary.json",dict(version=VERSION,
            status="complete" if complete else "blocked",failure=self.failure,
            physical_rows=self.physical,control_windows=self.controls,
            initialized=self.initialized,new_physics_steps_by_observer=0,ppo_updates=0,
            training_reward=None,teacher_admission=False,native_qualification=False,
            original_R0_terms_unchanged=True,legacy_physics_actions_unchanged=True))

def run_existing_app(*args,**kwargs):
    session=LiveSession(kwargs.get("project",ROOT),kwargs["output"])
    original_build=builder.build_reset_owned
    original_write=parameter.write_reset
    class ObservedJournal(Journal):
        def record(self,row):
            super().record(row) # Persist original physics first, even on observer failure.
            session.record(row)
        def begin(self,pre_action_state,action):
            super().begin(pre_action_state,action)
            session.begin(pre_action_state,action)
        def end(self,control_result):
            super().end(control_result)
            session.end(control_result)
    def build(sim,owners,**build_kwargs):
        try:
            result=original_build(sim,owners,**build_kwargs)
            session.native_refs.update(sim=weakref.ref(sim),rod=weakref.ref(owners.resources["runtime"]))
            return result
        finally:sim=owners=build_kwargs=None
    def reset_write(path,packet,error,identity):
        try:
            original_write(path,packet,error,identity)
            session.initialize(packet,error,identity,original_write)
        finally:session.native_refs.clear()
    try:
        with bindings((builder,"build_reset_owned",build),
            (parameter,"write_reset",reset_write),(evidence,"DualHandEvidenceJournal",ObservedJournal)):
            return original_run(*args,**kwargs)
    except BaseException as exc:
        session.failure=type(exc).__name__+": "+str(exc)
        raise
    finally:
        session.close()
        args=kwargs=original_build=original_write=None
