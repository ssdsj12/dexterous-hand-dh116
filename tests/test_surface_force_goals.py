"""The teacher explores legal force targets; normal force is never fixed5N."""
import importlib
import importlib.util
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_force_goals'
    assert importlib.util.find_spec(name) is not None,'force goal exploration missing'
    return importlib.import_module(name)


def test_initial_teacher_targets_span_multiple_forces_and_stay_legal():
    explorer=module().ForceGoalExplorer(8,'cpu',seed=12)
    goal=explorer.step(1/60)
    assert goal.shape==(8,2) and (goal>=2).all() and (goal<=12).all()
    assert goal.std()>.5 and not torch.allclose(goal,torch.full_like(goal,5.))


def test_force_targets_keep_exploring_and_follow_a_bounded_slew():
    e=module().ForceGoalExplorer(4,'cpu',seed=19);first=e.goal.clone();last=first.clone()
    for _ in range(180):
        goal=e.step(1/60)
        assert ((goal-last).abs()<=1/60+1e-12).all()
        last=goal.clone()
    assert (last-first).abs().max()>.05


def test_force_action_encodes_the_current_explored_normal_target():
    e=module().ForceGoalExplorer(3,'cpu',seed=8);goal=e.step(1/60)
    torch.testing.assert_close(2+(e.actions()+1)*5,goal)


def test_seeded_search_is_reproducible_and_independent_per_hand():
    a=module().ForceGoalExplorer(4,'cpu',seed=5);b=module().ForceGoalExplorer(4,'cpu',seed=5)
    torch.testing.assert_close(a.step(1/60),b.step(1/60))
    assert not torch.equal(a.goal[:,0],a.goal[:,1])


def test_pinching_keeps_the_chosen_force_without_further_target_exploration():
    e=module().ForceGoalExplorer(4,'cpu',seed=5)
    for _ in range(90):e.step(1/60)
    chosen=e.goal.clone();e.hold()
    for _ in range(600):
        torch.testing.assert_close(e.step(1/60),chosen,atol=0.,rtol=0.)
        torch.testing.assert_close(2+(e.actions()+1)*5,chosen)


def test_initial_low_force_search_still_chooses_distinct_commands():
    e=module().ForceGoalExplorer(8,'cpu',seed=12,initial_max_n=3.)
    assert (e.goal>=2.).all() and (e.goal<=3.).all() and e.goal.std()>.05
    e.hold();torch.testing.assert_close(e.step(1/60),e.goal)
