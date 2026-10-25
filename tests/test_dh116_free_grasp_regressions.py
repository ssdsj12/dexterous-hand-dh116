import numpy as np
from test_dh116_batched_task import make_env
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask


def test_warning_diagnostics_outside_geometry_subset_do_not_crash():
    envs=[make_env(0.) for _ in range(2)]
    task=BatchTask(envs)
    measured=task.stack_measurements([e.backend.measure() for e in envs])
    measured['_diagnostics']={1:[{'command_rad':[0.]*6}]}
    measured['_lowest_links']={0:'left/finger14_link'}
    task.consume(measured,np.ones(2,dtype=bool),1/480)
    assert envs[1].measurement['diagnostics']==measured['_diagnostics'][1]


def test_coupling_recovery_stays_inside_envelope_with_scalar_batch_parity():
    envs=[make_env(0.) for _ in range(2)]
    for e in envs:
        q=e.reference.copy()
        e.joint_envelope=dict(version='dh116-joint-envelope-v1',lower_rad=(q-.01).tolist(),
                              center_rad=q.tolist(),upper_rad=(q+.01).tolist())
        e.command=q+.01
        e.backend.q=e.command.copy()
        original=e.backend.measure
        def measured(original=original):
            m=original();m['mimic_errors'][0]=.005;return m
        e.backend.measure=measured
        e._measure()
    scalar,batched=envs
    action=np.ones(20)
    before=scalar.command.copy()
    scalar.step(action)
    task=BatchTask([batched])
    def advance(offset,command,allowed,live):
        batched.backend.advance(offset[0],command[0],palm_motion_allowed=bool(allowed[0]),
            palm_action=task.palm_actions[0],grip_action=np.ones(2))
        return task.stack_measurements([batched.backend.measure()])
    task.step(action[None],advance)
    for e in envs:
        assert np.all(e.command<=np.array(e.joint_envelope['upper_rad'])+1e-12)
        assert np.max(abs(e.command-before))<=.05/60+1e-12
    np.testing.assert_allclose(scalar.command,batched.command,atol=1e-12)
