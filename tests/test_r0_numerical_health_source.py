"""CPU exact-runtime/SDK-shaped doubles; never imports or runs a native solver."""
import gc
from copy import deepcopy
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
from test_flexible_parameter_readback import case
from test_r0_health_separation import api,exchange
from test_dh116_deformable_control_step import inputs


def capture(c):
    return api().capture_solver_health(c.runtime,backend=c.backend,sim=c.sim,expected_clock=c.expected)


@pytest.fixture
def health_case(case):
    c=case
    c.backend.__file__=str(api().BACKEND)
    c.base.physics_step=3;c.base.rod_tick=36;c.rope.time=3/480
    c.sim.current_time_step_index=3;c.sim.current_time=3/480
    c.expected=c.runtime.clock()
    return c


def row_for(c):
    task,_,_=inputs();_,row=exchange(task,3)
    clock=dict(c.expected,policy_step=0,substep=0)
    row["rod"].update(c.expected)
    row["action_readback"]["post"]["identity"]=[clock]
    row["post_step_clearance"]["identity"]=clock
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import VERSION as PARAM_VERSION
    row["material_director"]=dict(version=PARAM_VERSION,identity=dict(c.expected),
        source_instance_id=id(c.rope),sources=dict(backend_sha256=api().BACKEND_SHA))
    row["director_capture_error"]=None
    return row


@pytest.mark.parametrize("code",[0,4])
def test_actual_runtime_getter_zero_and_nonzero_are_distinct_owned_samples(health_case,monkeypatch,code):
    c=health_case;events=[]
    monkeypatch.setattr(type(c.rope),"last_error",property(lambda self:events.append("getter") or code))
    status=capture(c)
    assert events and c.events==[]
    assert status["last_error"]==code and status["identity"]==c.expected
    assert status["source_scope"]=="native_getter"
    assert status["native_readback_qualified"] is False
    assert status["new_physics_steps"]==status["new_solver_steps"]==0
    report=api().observe_health(row_for(c),solver_status=status)
    assert report["numerical_health"]["value"] is (code==0)
    assert report["original_stable"] is report["motion"]["hold_motion"] is None
    assert report["native_qualification"] is report["teacher_admission"] is False
    assert c.runtime.clock()==c.expected


@pytest.mark.parametrize("value",[None,False,"",0.,float("nan")])
def test_missing_or_wrong_getter_type_stays_unknown(health_case,monkeypatch,value):
    c=health_case
    def read(self):
        if value is None:raise AttributeError("getter unavailable")
        return value
    monkeypatch.setattr(type(c.rope),"last_error",property(read))
    status=capture(c)
    assert status["last_error"] is None and status["capture_error"]
    report=api().observe_health(row_for(c),solver_status=status)
    assert report["numerical_health"]==dict(value=None,source=None,
        definition="finite_recorded_endpoints_and_explicit_native_error_zero")


@pytest.mark.parametrize("damage",["clock","backend","owner"])
def test_getter_cannot_change_clock_backend_or_owner_between_reads(health_case,monkeypatch,damage):
    c=health_case
    def read(self):
        if damage=="clock":c.base.physics_step+=1
        elif damage=="backend":c.backend.__file__="/wrong/ropesim.so"
        else:c.backend.RopeBatch=object
        return 0
    monkeypatch.setattr(type(c.rope),"last_error",property(read))
    with pytest.raises(ValueError):capture(c)


def test_caller_supplied_native_scope_is_not_a_getter_observation(health_case):
    c=health_case
    with pytest.raises(ValueError,match="CPU"):
        api().make_solver_status(c.expected,0,source_scope="native_getter",
            source_instance_id=id(c.rope),backend_sha256=api().BACKEND_SHA)


def test_reader_does_not_keep_native_owner_through_failure_traceback(health_case):
    c=health_case;ref=weakref.ref(c.rope);runtime_ref=weakref.ref(c.runtime)
    c.backend.__file__="/wrong/backend"
    with pytest.raises(ValueError) as caught:capture(c)
    tb=caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name=="capture_solver_health":
            assert all(tb.tb_frame.f_locals.get(k) is None for k in ("runtime","backend","sim","rope","boundary"))
        tb=tb.tb_next
    c.runtime=None;c.base=None;c.rope=None
    gc.collect()
    assert ref() is runtime_ref() is None


@pytest.mark.parametrize("field",["node_positions","velocities"])
def test_no_joint_guard_does_not_make_nonfinite_rod_healthy(health_case,field):
    c=health_case;row=row_for(c);row["rod"][field][0,0,0]=np.nan
    report=api().observe_health(row,solver_status=capture(c))
    assert row["guard_failure"] is None and report["numerical_health"]["value"] is False
    assert report["constraint_health"]["value"] is None


def test_packet_validation_recomputes_source_and_never_trusts_healthy_boolean(health_case):
    c=health_case;row=row_for(c);packet=api().observe_health(row,solver_status=capture(c))
    assert api().validate_health(packet,row)==packet
    row["rod"]["velocities"][0,0,0]=.7
    with pytest.raises(ValueError,match="recomputed"):api().validate_health(packet,row)


def test_native_status_from_another_rope_cannot_qualify_this_exchange(health_case):
    c=health_case;row=row_for(c);status=capture(c)
    row["material_director"]["source_instance_id"]+=1
    with pytest.raises(ValueError,match="owner"):api().observe_health(row,solver_status=status)


def test_missing_actual_owner_anchor_keeps_native_getter_unknown(health_case):
    c=health_case;row=row_for(c);status=capture(c)
    del row["material_director"]
    result=api().observe_health(row,solver_status=status)
    assert result["numerical_health"]["value"] is None
    assert result["solver_owner_bound"] is False


@pytest.mark.parametrize("field",["forces","torques","rod_force"])
def test_nonfinite_actual_wrench_is_not_healthy_or_constraint_safe(health_case,field):
    c=health_case;row=row_for(c)
    if field=="rod_force":row["rod"]["distributed_contact"][0,0,0]=np.nan
    else:row["rod"]["hand_wrenches_com_world"]["left"][field][0,0,0]=np.nan
    result=api().observe_health(row,solver_status=capture(c))
    assert result["numerical_health"]["value"] is False
    assert result["constraint_health"]["value"] is None


@pytest.fixture(scope="module")
def projected_native_rows():
    """Read only the frozen native ledger; never synthesize a missing getter."""
    import hashlib,json
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    receipt=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
        "flex.f8-original-projected-pinch-native/"
        "d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58/"
        "attempt-zy2fao0h/receipt.json")
    data=receipt.read_bytes()
    assert hashlib.sha256(data).hexdigest()=="04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3"
    path=Path(json.loads(data)["output"])/"physical.jsonl"
    digest=hashlib.sha256();selected={};physical_count=0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            assert line.endswith(b"\n")
            row=json.loads(line)
            if row.get("operation")!="physical":continue
            physical_count+=1
            if physical_count+2 in (3,176,234):
                exchange_row=decode_owned(row["payload"])["exchange"]
                selected[exchange_row["physics_step"]]=exchange_row
    assert digest.hexdigest()=="9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4"
    assert physical_count==232 and set(selected)=={3,176,234}
    return selected


@pytest.mark.parametrize("step",[3,176,234])
def test_frozen_actual_ledger_shapes_guard_and_missing_getter_remain_unknown(projected_native_rows,step):
    row=projected_native_rows[step]
    for side in ("left","right"):
        for name in ("forces","torques"):
            assert row["rod"]["hand_wrenches_com_world"][side][name].shape==(1,5,3)
    result=api().observe_health(row,solver_status=None)
    assert result["numerical_health"]["value"] is None
    assert result["solver_status"] is None and result["solver_owner_bound"] is False
    assert all(result["finite_channels"].values())
    assert result["constraint_health"]["value"] is True
    assert result["constraint_health"]["source_error"] is None
    assert result["physical_terminal"]["reason"] is None
    assert result["motion"]["hold_motion"] is None and result["original_stable"] is None
    assert result["new_physics_steps"]==result["new_solver_steps"]==0
    assert result["native_qualification"] is result["teacher_admission"] is False


@pytest.mark.parametrize("field",["source_sha256","sdk_time_s","guard_failure"])
def test_invalid_constraint_source_cannot_become_unknown_but_reward_usable(health_case,field):
    c=health_case;row=row_for(c)
    if field=="guard_failure":row["original_guard_failure"]="excess_force"
    elif field=="sdk_time_s":row["post_step_clearance"][field]+=1.
    else:row["post_step_clearance"][field]={}
    with pytest.raises(ValueError):
        api().observe_health(row,solver_status=capture(c))


@pytest.mark.parametrize("field",["forces","torques"])
def test_actual_wrench_shape_cannot_drop_the_environment_axis(health_case,field):
    c=health_case;row=row_for(c)
    row["rod"]["hand_wrenches_com_world"]["left"][field]=np.zeros((5,3))
    with pytest.raises(ValueError,match="shape"):
        api().observe_health(row,solver_status=capture(c))
