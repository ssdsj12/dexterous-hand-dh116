import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_vertical_force import object_vertical_components, VerticalForceStats


def test_action_reaction_sums_signed_normal_and_friction_without_norms():
    n=torch.zeros(2,2,3,3,dtype=torch.float64);f=n.clone()
    n[0,0,:,2]=torch.tensor([-1.,.5,-.3]);n[0,1,0,2]=.2
    f[0,0,0,2]=-.4;f[0,1,0,2]=-.6
    n[...,0]=1000.  # Horizontal squeeze must not count as upward support.
    actual=object_vertical_components(n,f)
    assert actual.shape==(2,2,2)
    assert torch.allclose(actual[0],torch.tensor([[.8,.4],[-.2,.6]],dtype=torch.float64))
    assert not actual[1].any()


def test_statistics_exclude_inactive_rows_and_distinguish_peak_from_mean():
    stats=VerticalForceStats('cpu')
    v=torch.tensor([[[.5,.2],[.3,.4]],[[100.,0],[100.,0]]],dtype=torch.float64)
    stats.add(v,torch.tensor([1.18,1.18]),torch.tensor([True,False]))
    stats.add(v*0,torch.tensor([1.18,1.18]),torch.tensor([True,False]))
    row=stats.summary()
    assert row['hands_total_mean_n']==pytest.approx(.4)
    assert row['hands_total_peak_n']==pytest.approx(.8)
    assert row['above_weight_sample_fraction']==0.
    assert row['sample_count']==2
    assert row['hand_total_mean_n']==pytest.approx([.25,.15])
    stats.clear();assert stats.summary()['sample_count']==0


def test_nonfinite_active_measurement_is_not_silently_replaced():
    stats=VerticalForceStats('cpu')
    with pytest.raises(RuntimeError,match='nonfinite'):
        stats.add(torch.full((1,2,2),float('nan')),torch.tensor([1.18]),torch.tensor([True]))


def test_directional_means_split_samples_before_averaging_and_reset():
    stats=VerticalForceStats('cpu')
    v=torch.zeros((3,2,2),dtype=torch.float64)
    # First sample: opposing hands cancel partially (+3 -1 = +2).
    # Second sample: total -4. Third inactive outlier must never count.
    v[:,0,0]=torch.tensor([3.,-4.,1000.]);v[0,1,0]=-1.
    stats.add(v,torch.ones(3),torch.tensor([True,True,False]))
    row=stats.summary()
    assert row['upward_mean_n']==pytest.approx(1.)
    assert row['downward_mean_n']==pytest.approx(2.)
    assert row['upward_peak_n']==pytest.approx(2.)
    assert row['downward_peak_n']==pytest.approx(4.)
    assert row['upward_mean_n']-row['downward_mean_n']==pytest.approx(row['hands_total_mean_n'])
    stats.clear();row=stats.summary()
    assert row['upward_mean_n']==0 and row['downward_mean_n']==0
    assert row['upward_peak_n'] is None and row['downward_peak_n'] is None


def test_telemetry_contract_excludes_friction_while_preserving_separate_diagnostics():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import CONTRACT, REWARD_CONTRACT
    stats=VerticalForceStats('cpu')
    stats.add(torch.tensor([[[.4,-1.],[.6,-2.]]],dtype=torch.float64),
              torch.ones(1),torch.tensor([True]))
    row=stats.summary()
    assert row['version']=='normal_vertical_force_v3'
    assert row['source']=='object_filtered_normal_on_hand_negated'
    assert row['friction_excluded'] is True
    assert row['legacy_total_fields']=='signed_normal_z_only'
    assert row['hand_friction_mean_n']==pytest.approx([-1.,-2.])
    assert row['hands_total_mean_n']==pytest.approx(1.)
    assert row['downward_mean_n']==pytest.approx(0.)
    assert REWARD_CONTRACT['friction_excluded'] is True
    assert REWARD_CONTRACT['source']=='object_filtered_normal_on_hand_negated'


@pytest.mark.parametrize('kind',['rollout','lift_entry'])
def test_normal_force_drives_means_peaks_and_weight_comparison_when_friction_opposes(kind):
    from go2_pvcnn.control.dh116.tabletop_vertical_force import (
        LiftEntryForceStats,write_directional_force_scalars)
    stats=(VerticalForceStats if kind=='rollout' else LiftEntryForceStats)('cpu')
    components=torch.tensor([[[.4,-1.],[.6,-2.]],[[-2.,4.],[-2.,6.]],
                             [[100.,100.],[100.,100.]]],dtype=torch.float64)
    stats.add(components,torch.full((3,),.5),torch.tensor([True,True,False]))
    row=stats.summary()
    assert row['hand_total_mean_n']==pytest.approx([-.8,-.7])
    assert row['hand_total_mean_n']==row['hand_normal_mean_n']
    assert row['hands_total_mean_n']==pytest.approx(-1.5)
    assert row['hand_friction_mean_n']==pytest.approx([1.5,2.])
    assert row['upward_mean_n']==pytest.approx(.5)
    assert row['downward_mean_n']==pytest.approx(2.)
    assert row['upward_peak_n']==pytest.approx(1.)
    assert row['downward_peak_n']==pytest.approx(4.)
    assert row['hands_total_peak_n']==pytest.approx(1.)
    assert row['above_weight_sample_fraction']==pytest.approx(.5)
    assert row['force_minus_weight_mean_n']==pytest.approx(-2.)
    class Writer:
        def __init__(self):self.points={}
        def add_scalar(self,tag,value,step):self.points[tag]=(value,step)
    writer=Writer()
    write_directional_force_scalars(writer,row,7)
    assert writer.points['rollout_force_N/向上力_平均']==(pytest.approx(.5),7)
    assert writer.points['rollout_force_N/向下力_平均']==(pytest.approx(2.),7)
    assert writer.points['rollout_force_N/向上力_瞬时峰值']==(pytest.approx(1.),7)
    assert writer.points['rollout_force_N/向下力_瞬时峰值']==(pytest.approx(4.),7)
    assert writer.points['rollout_force_N/竖直合力_向上为正']==(pytest.approx(-1.5),7)
