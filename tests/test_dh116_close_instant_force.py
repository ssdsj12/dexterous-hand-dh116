"""Close completion reads the current normal force after mechanical dwell."""
import pytest
import torch

from test_dh116_close_downward_penalty import task
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES


@pytest.mark.parametrize('hz', [60, 480])
@pytest.mark.parametrize('current_force', [0., 1.])
def test_negative_force_preserves_capped_dwell_then_current_force_allows_close(hz, current_force):
    t = task(1)
    t.begin_step(torch.zeros((1, 20)))
    m = loads(t, 4.)
    for _ in range(hz):
        t.consume(m, ~t.done, 1 / hz, vertical_force_n=torch.tensor([-1.]))
    assert t.phase.item() == 1
    assert t.failure.item() == 0
    assert t.contact_time.item() == pytest.approx(.2)
    assert t.close_downward_impulse.item() == pytest.approx(1.)
    assert t.terms['downward_force'].item() == pytest.approx(-.5)

    t.consume(m, ~t.done, 1 / hz, vertical_force_n=torch.tensor([current_force]))
    assert t.phase.item() == 3
    assert t.contact_time.item() == 0.
    assert t.close_downward_impulse.item() == pytest.approx(1.)
    assert t.terms['close_downward'].item() == 0.
    assert t.terms['grasp_ready'].item() == (3. if current_force > 0. else 0.)
    assert t.observe().shape == (1, 151)


@pytest.mark.parametrize('hz', [60, 480])
def test_negative_completion_sample_blocks_without_erasing_previous_positive_dwell(hz):
    t = task(1)
    m = loads(t, 4.)
    dt = 1 / hz
    for _ in range(round(.2 * hz) - 1):
        t.consume(m, ~t.done, dt, vertical_force_n=torch.ones(1))
    assert t.phase.item() == 1
    t.consume(m, ~t.done, dt, vertical_force_n=-torch.ones(1))
    assert t.phase.item() == 1
    assert t.contact_time.item() == pytest.approx(.2)
    t.consume(m, ~t.done, dt, vertical_force_n=torch.zeros(1))
    assert t.phase.item() == 3


@pytest.mark.parametrize('instability', ['moving', 'weak_load', 'slipping'])
def test_nonforce_instability_resets_dwell_before_nonnegative_sample(instability):
    t = task(1)
    m = loads(t, 4.)
    t.grip_valid[:] = True
    t.grip.copy_(m['object_relative'])
    for _ in range(20):
        t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(1))
    assert t.contact_time.item() == pytest.approx(.2)

    unstable = {key: value.clone() for key, value in m.items()}
    if instability == 'moving':
        unstable['object_velocity'][:, 0] = .021
    elif instability == 'weak_load':
        unstable = loads(t, 1.8)
    else:
        unstable['object_relative'][:, 0] += .004
    t.consume(unstable, ~t.done, .01, vertical_force_n=-torch.ones(1))
    assert t.phase.item() == 1
    assert t.contact_time.item() == 0.

    for _ in range(19):
        t.consume(m, ~t.done, .01, vertical_force_n=torch.zeros(1))
    assert t.phase.item() == 1
    t.consume(m, ~t.done, .01, vertical_force_n=torch.zeros(1))
    assert t.phase.item() == 3


def test_capped_dwell_waits_only_until_unchanged_four_second_timeout():
    t = task(1)
    m = loads(t, 4.)
    for _ in range(3):
        t.consume(m, ~t.done, .1, vertical_force_n=-torch.ones(1))
    assert t.contact_time.item() == pytest.approx(.2)
    t.elapsed[:] = 3.99
    t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(1))
    assert t.elapsed.item() == pytest.approx(4.)
    assert t.failure.item() == 0
    t.consume(m, ~t.done, .01, vertical_force_n=torch.zeros(1))
    assert t.failure.item() == FAILURE_CODES['grasp_not_established']
    assert t.phase.item() == 1


def test_current_normal_force_decides_close_even_when_combined_lift_force_disagrees():
    t = task(2)
    m = loads(t, 4.)
    for _ in range(20):
        t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(2),
                  lift_vertical_force_n=torch.ones(2))
    assert t.phase.tolist() == [1, 1]
    t.consume(m, ~t.done, .01, vertical_force_n=torch.tensor([-1., 0.]),
              lift_vertical_force_n=torch.tensor([10., -10.]))
    assert t.phase.tolist() == [1, 3]
    assert t.contact_time.tolist() == pytest.approx([.2, 0.])


def test_inactive_failed_and_nonfinite_samples_cannot_complete_waiting_close():
    t = task(5)
    m = loads(t, 4.)
    for _ in range(20):
        t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(5))
    assert t.contact_time.tolist() == pytest.approx([.2] * 5)
    t.failure[1] = FAILURE_CODES['invalid_action']
    m['clearance'][2] = -.002
    t.consume(m, torch.tensor([False, True, True, True, True]), .01,
              vertical_force_n=torch.tensor([0., 0., 0., float('nan'), 0.]))
    assert t.phase.tolist() == [1, 1, 1, 1, 3]
    assert torch.isfinite(t.reward).all()


def test_legacy_hold_accumulates_its_own_dwell_independently_of_force_sign():
    t = task(1, skip_hold=False, close_downward_penalty=False)
    m = loads(t, 4.)
    for _ in range(20):
        t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(1))
    assert t.phase.item() == 1
    t.consume(m, ~t.done, .01, vertical_force_n=torch.zeros(1))
    assert t.phase.item() == 2
    assert t.contact_time.item() == 0.
    for _ in range(40):
        t.consume(m, ~t.done, .01, vertical_force_n=-torch.ones(1))
    assert t.phase.item() == 2
    assert t.contact_time.item() == pytest.approx(.3)
    t.consume(m, ~t.done, .01, vertical_force_n=torch.zeros(1))
    assert t.phase.item() == 3


def test_lift_stability_keeps_using_normal_plus_friction_force():
    t = task(2)
    t.phase[:] = 3
    m = loads(t, 4.)
    m['bottom'][:] = .33
    for _ in range(50):
        t.consume(m, ~t.done, .01, vertical_force_n=torch.tensor([-1., 1.]),
                  lift_vertical_force_n=torch.tensor([1., -1.]))
    assert t.success.tolist() == [True, False]


def test_completion_contract_identifies_current_normal_force_and_independent_dwell():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_COMPLETION_FORCE_CONTRACT
    assert CLOSE_COMPLETION_FORCE_CONTRACT['version'] == 'instantaneous_close_completion_force_v1'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['force_source'] == 'vertical_force_n'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['force_channel'] == 'hand_object_normal_only'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['sample'] == 'current_physics_substep'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['condition'] == 'vertical_force_n >= 0'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['timing'] == 'after_nonforce_readiness_dwell'
    assert CLOSE_COMPLETION_FORCE_CONTRACT['nonforce_close_dwell_s'] == .2
    assert CLOSE_COMPLETION_FORCE_CONTRACT['nonforce_hold_dwell_s'] == .3
    assert CLOSE_COMPLETION_FORCE_CONTRACT['negative_force'] == 'blocks_transition_without_resetting_nonforce_dwell'
