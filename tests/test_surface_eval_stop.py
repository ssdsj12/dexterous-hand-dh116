import torch
from go2_pvcnn.control.dh116 import tabletop_surface_student_evaluation as evaluation


def test_success_stop_only_counts_new_native_successes():
    before=torch.tensor([4,2])
    assert evaluation.evaluation_stop(torch.tensor([1,0]),torch.tensor([0,0]),before,before,
        first_episode_only=False,stop_on_success=True) is None
    assert evaluation.evaluation_stop(torch.tensor([1,0]),torch.tensor([0,0]),before+torch.tensor([0,1]),before,
        first_episode_only=False,stop_on_success=True)=='native_success_observed'


def test_first_episode_mode_keeps_running_until_every_row_has_ended():
    z=torch.zeros(2,dtype=torch.long)
    assert evaluation.evaluation_stop(torch.tensor([1,0]),z,z,z,first_episode_only=True) is None
    assert evaluation.evaluation_stop(torch.ones(2,dtype=torch.long),z,z,z,
        first_episode_only=True)=='all_envs_completed_first_episode'
