"""Read-only B40/B41 owner observer for the unchanged 371D contract.

The caller runs the original controller. This observer copies its memory and
checks reset/begin/end lifecycle consistency; it never writes targets, advances
physics, constructs a policy or grants native/reset/R0 qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import tabletop_teacher_preaction_readback as base
from .tabletop_teacher_mounted_wrist_source import _clock, RESET_KEYS
from .tabletop_tensor_task import TensorTask
from .tabletop_surface_contract import ACTION_CONTRACT
from .linear_coupling import active_limits

VERSION = "dh116-teacher-policy-owner-source-v1"
SOURCE = "original_tensortask_command_actions_at_pre_action_boundary"
SCOPE = "controller_memory_lifecycle_consistency_not_native_execution"
LOW, HIGH = [np.tile(v, 2) for v in active_limits()]


def _implementation():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in (
        Path(__file__).name, "tabletop_tensor_task.py", "tabletop_free_wrist_grasp.py",
        "tabletop_surface_contract.py", "tabletop_teacher_preaction_readback.py",
        "tabletop_teacher_mounted_wrist_source.py", "linear_coupling.py")}


def _action_sha():
    return hashlib.sha256(json.dumps(ACTION_CONTRACT,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _array(value, shape):
    result = np.array(value,dtype=np.float64,copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"finite controller source {shape} required")
    return result


def _state(state):
    command,actions = _array(state["command"],(12,)),_array(state["actions"],(24,))
    if (np.any(command < LOW) or np.any(command > HIGH) or np.any(abs(actions)>1)
            or state.get("dtype") != "float64"):
        raise ValueError("original float64 command limits and accepted action24 required")
    return command,actions


def _same_epoch(left,right):
    return all(left[key]==right[key] for key in RESET_KEYS)


def _completed(before,after):
    if (not _same_epoch(before,after) or after["policy_step"]!=before["policy_step"]+1
            or after["physics_step"]!=before["physics_step"]+8
            or after["substep"]!=0 or after["rod_tick"]<=before["rod_tick"]
            or after["solver_time_s"]<=before["solver_time_s"]):
        raise ValueError("complete next eight-step control boundary required")


def _checked(packet,expected):
    if (not isinstance(packet,dict) or packet.get("version")!=VERSION
            or packet.get("source")!=SOURCE or packet.get("scope")!=SCOPE
            or packet.get("implementation_sha256")!=_implementation()
            or packet.get("action_contract_sha256")!=_action_sha()
            or packet.get("joint_order")!=ACTION_CONTRACT["joint_order"]
            or packet.get("native_qualification") is not False):
        raise ValueError("exact controller owner source/implementation required")
    current=_clock(packet["identity"])
    if current!=expected or current["substep"]!=0:
        raise ValueError("same pre-action control-boundary identity required")
    command,actions=_state(packet["state"])
    reset=packet["reset"]
    reset_clock=_clock(reset["identity"])
    reset_command,reset_actions=_state(reset["state"])
    seed_command,seed_actions=_state(reset["seed"])
    if (not _same_epoch(reset_clock,current) or reset_clock["policy_step"]!=0
            or reset_clock["substep"]!=0 or not np.array_equal(reset_command,seed_command)
            or not np.array_equal(reset_actions,seed_actions) or np.any(seed_actions!=0)):
        raise ValueError("explicit original reset seed parity required")
    submission=packet["submission"]
    if submission is None:
        if (current!=reset_clock or not np.array_equal(command,reset_command)
                or not np.array_equal(actions,reset_actions)):
            raise ValueError("unrecorded reset/controller transition")
    else:
        previous_clock=_clock(submission["identity"])
        if previous_clock["physics_step"] != reset_clock["physics_step"]+8*previous_clock["policy_step"]:
            raise ValueError("submission clock does not follow the recorded reset boundary")
        _completed(previous_clock,current)
        previous_command,previous_actions=_state(submission["before"])
        accepted_command,accepted_actions=_state(submission["accepted"])
        if (previous_clock["policy_step"]<reset_clock["policy_step"]
                or not np.array_equal(accepted_command,previous_command)
                or not np.array_equal(_array(submission["old_command"],(12,)),previous_command)
                or not np.array_equal(_array(submission["previous_actions"],(24,)),previous_actions)
                or not np.array_equal(_array(submission["last_raw_action24"],(24,)),accepted_actions)
                or not np.array_equal(actions,accepted_actions)
                or not np.array_equal(_array(packet["end_step_previous"],(12,)),command)):
            raise ValueError("recorded begin/end_step action/command lineage differs")
    return command,actions


class PolicyOwnerSource:
    """One selected environment of an actual TensorTask, observed only.

    record_reset follows the caller's original reset; capture_pre_action precedes
    begin_step; record_submission follows accepted begin_step before control or
    target writes. Next capture requires the complete next control boundary.
    Clock truth, actual reset execution and physical/R0 coverage remain external.
    """
    def __init__(self,task,*,read_identity,data_env_index,expected_env_id):
        if (type(data_env_index) is not int or data_env_index<0
                or type(expected_env_id) is not int or expected_env_id<0):
            raise ValueError("explicit controller data row and environment identity required")
        self.task,self.read_identity=task,read_identity
        self.row,self.env_id=data_env_index,expected_env_id
        self._reset=self._ready=self._pending=self._last_submission=None
        self._check_task()

    def _check_task(self):
        task=self.task
        if type(task) is not TensorTask:
            raise ValueError("actual original TensorTask owner required, not SDK/exchange targets")
        if (type(task.n) is not int or self.row>=task.n or task.action_size!=24
                or any(getattr(task,k,None) is not True for k in (
                    "free_wrist_grasp","surface_force_sharing","continuous_motion"))
                or any(getattr(getattr(task,k),"__func__",None) is not getattr(TensorTask,k)
                       for k in ("begin_step","end_step","reset"))):
            raise ValueError("original surface action24 TensorTask lifecycle required")
        for name,expected in (("low",LOW),("high",HIGH)):
            value=getattr(task,name)
            if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float64
                    or not np.array_equal(base._owned(value),expected)):
                raise ValueError("original active joint order/limits required")

    def _row(self,name,size):
        value=getattr(self.task,name,None)
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float64
                or tuple(value.shape)!=(self.task.n,size)):
            raise ValueError(f"actual float64 controller {name} buffer required")
        return _array(base._owned(value)[self.row],(size,)).tolist()

    def _identity(self):
        identity=_clock(self.read_identity())
        if identity["env_id"]!=self.env_id or identity["substep"]!=0:
            raise ValueError("selected environment pre-action boundary required")
        return identity

    def _snapshot(self,*,submission=False):
        self._check_task()
        before=self._identity()
        state=dict(command=self._row("command",12),actions=self._row("actions",24),dtype="float64")
        _state(state)
        if bool(self.task.done[self.row]):
            raise ValueError("nonterminal accepted controller state required")
        result=dict(identity=before,state=state,previous=self._row("previous",12))
        if submission:
            result.update(old_command=self._row("old_command",12),
                previous_actions=self._row("previous_actions",24),
                last_raw_action24=self._row("last_raw_action24",24))
        if self._identity()!=before:
            raise ValueError("clock changed during controller copy")
        return result

    def _seed(self,name,size):
        per_environment=getattr(self.task,"_per_env_reset_seed",{})
        if name in per_environment:
            value=per_environment[name]
            if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(self.task.n,size):
                raise ValueError("complete per-environment reset seed required")
            value=value[self.row]
        else:
            value=getattr(self.task,"_seed",{}).get(name)
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float64
                or tuple(value.shape)!=(size,)):
            raise ValueError("explicit original controller reset seed required")
        return _array(base._owned(value),(size,)).tolist()

    def record_reset(self):
        snapshot=self._snapshot()
        identity=snapshot["identity"]
        if (identity["policy_step"]!=0 or (self._reset is not None and (
                identity["run_uuid"]!=self._reset["identity"]["run_uuid"]
                or identity["reset_id"]<=self._reset["identity"]["reset_id"]
                or identity["episode_uid"]==self._reset["identity"]["episode_uid"]))):
            raise ValueError("new explicit reset/episode identity required")
        seed=dict(command=self._seed("command",12),actions=self._seed("actions",24),dtype="float64")
        command,actions=_state(seed)
        if (snapshot["state"]!=seed or np.any(actions!=0)
                or self._identity()!=identity):
            raise ValueError("original reset seed/state parity required")
        self._reset=dict(identity=identity,state=snapshot["state"],seed=seed)
        self._ready=self._pending=self._last_submission=None
        return deepcopy(self._reset)

    def capture_pre_action(self):
        if self._reset is None:
            raise ValueError("record_reset required before capture")
        snapshot=self._snapshot()
        identity=snapshot["identity"]
        if not _same_epoch(identity,self._reset["identity"]):
            raise ValueError("reset/episode/rod identity changed without record_reset")
        if self._pending is not None:
            _completed(self._pending["identity"],identity)
            if snapshot["state"]["actions"]!=self._pending["accepted"]["actions"]:
                raise ValueError("last accepted action changed without submission")
            if snapshot["previous"]!=snapshot["state"]["command"]:
                raise ValueError("original end_step previous=command update missing")
        elif self._ready is not None:
            if identity!=self._ready["identity"] or snapshot["state"]!=self._ready["state"]:
                raise ValueError("unrecorded controller transition")
        elif identity!=self._reset["identity"] or snapshot["state"]!=self._reset["state"]:
            raise ValueError("reset state changed before first pre-action capture")
        packet=dict(version=VERSION,source=SOURCE,scope=SCOPE,identity=identity,state=snapshot["state"],
            reset=deepcopy(self._reset),submission=deepcopy(self._last_submission),
            end_step_previous=snapshot["previous"],implementation_sha256=_implementation(),
            action_contract_sha256=_action_sha(),joint_order=deepcopy(ACTION_CONTRACT["joint_order"]),
            native_qualification=False)
        _checked(packet,identity)
        self._ready=deepcopy(snapshot)
        self._pending=None
        return packet

    def record_submission(self):
        if self._ready is None or self._pending is not None:
            raise ValueError("one prior pre-action capture required per submission")
        snapshot=self._snapshot(submission=True)
        ready=self._ready
        if (snapshot["identity"]!=ready["identity"]
                or snapshot["state"]["command"]!=ready["state"]["command"]
                or snapshot["old_command"]!=ready["state"]["command"]
                or snapshot["previous_actions"]!=ready["state"]["actions"]
                or snapshot["last_raw_action24"]!=snapshot["state"]["actions"]):
            raise ValueError("accepted original begin_step side effects required before target writes")
        event=dict(identity=snapshot["identity"],before=deepcopy(ready["state"]),
            accepted=snapshot["state"],old_command=snapshot["old_command"],
            previous_actions=snapshot["previous_actions"],last_raw_action24=snapshot["last_raw_action24"])
        self._pending=deepcopy(event)
        self._last_submission=deepcopy(event)
        self._ready=None
        return deepcopy(event)


def adapt_policy_owner(packet,*,expected_identity):
    expected=_clock(expected_identity)
    command,actions=_checked(packet,expected)
    values=np.full(371,np.nan)
    valid=np.zeros(371,dtype=bool)
    sources=[None]*371
    values[203:215]=2*(command-LOW)/(HIGH-LOW)-1
    values[215:239]=actions
    valid[203:239]=True
    sources[203:215]=[SOURCE+":task.command"]*12
    sources[215:239]=[SOURCE+":task.actions"]*24
    return dict(version=VERSION,contract_sha256=base.CONTRACT_SHA256,identity=expected,
        values=values,valid=valid,sources=sources,source_packet=deepcopy(packet),
        qualified_for_teacher_ppo=False,new_physics_steps=0,optimizer_updates=0)


def validate_policy_owner(observation):
    if not isinstance(observation,dict) or observation.get("version")!=VERSION:
        raise ValueError("independent controller owner version required")
    expected=adapt_policy_owner(observation["source_packet"],expected_identity=observation["identity"])
    for key in ("contract_sha256","sources","qualified_for_teacher_ppo","new_physics_steps","optimizer_updates"):
        if observation.get(key)!=expected[key]:
            raise ValueError("controller observation differs from reconstruction")
    valid,values=observation.get("valid"),observation.get("values")
    if (not isinstance(valid,np.ndarray) or valid.dtype!=np.bool_
            or not isinstance(values,np.ndarray) or values.shape!=(371,)
            or not np.array_equal(valid,expected["valid"])
            or not np.array_equal(values,expected["values"],equal_nan=True)):
        raise ValueError("controller values/mask differ from reconstruction")
    return dict(valid_channels=36,unknown_channels=335,native_qualification=False)
