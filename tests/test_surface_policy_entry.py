from types import SimpleNamespace
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm
from go2_pvcnn.control.dh116 import tabletop_surface_exploration as mode


def test_24d_actor_has_no_force_bias_on_last_two_wrist_actions():
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.zeros(2,32)},batch_size=[2],device='cpu')
    algo=make_algorithm(obs,steps=8,action_size=24)
    assert algo.policy.actor[-2].out_features==24
    assert not algo.policy.actor[-2].bias.detach().any()


def test_surface_option_requires_free_continuous_wrist_policy():
    with pytest.raises(ValueError):
        mode.validate_options(SimpleNamespace(surface_force_sharing=True,free_wrist_grasp=False))


@pytest.mark.parametrize('surface,demonstrations', [(True,None),(False,'native.pt')])
def test_stable_hold_cannot_silently_run_without_native_surface_demonstrations(surface,demonstrations):
    args=SimpleNamespace(surface_force_sharing=surface,free_wrist_grasp=True,
        surface_imitation_mode='stable_hold_v2',surface_demonstrations=demonstrations)
    with pytest.raises(ValueError,match='stable'):
        mode.validate_options(args)


def test_native_batch_readback_is_explicitly_scoped_to_surface_mode():
    with pytest.raises(ValueError,match='native-sensor-batch'):
        mode.validate_options(SimpleNamespace(native_sensor_batch=True,surface_force_sharing=False))


def test_surface_option_forbids_silent_legacy_checkpoint_restore():
    args=SimpleNamespace(surface_force_sharing=True,free_wrist_grasp=True,initialize_from='old.pt',resume='')
    with pytest.raises(ValueError):mode.validate_options(args)


def packet():
    return dict(effective_pad_loads_n=torch.zeros(2,2,5),
        effective_pad_vectors_w=torch.zeros(2,2,5,3),effective_pad_points_w=torch.zeros(2,2,5,3),
        effective_pad_valid=torch.zeros(2,2,5,dtype=torch.bool),
        effective_pad_body_ids=torch.full((2,2,5),-1,dtype=torch.long),
        effective_pad_depth_m=torch.zeros(2,2,5),pad_reference_points_w=torch.zeros(2,2,5,3),
        rejected_contact_loads_n=torch.zeros(2,2),effective_pad_bad_env=torch.zeros(2,dtype=torch.bool))


def test_native_only_pad_fields_get_schema_validation_without_fake_host_reference():
    report=mode.validate_startup_measurements(packet(),2)
    assert report['effective_pad_loads_n']['shape']==[2,2,5]
    assert report['pad_reference_points_w']['evidence']=='FK_geometry_not_contact'


@pytest.mark.parametrize('key', ['effective_pad_loads_n','rejected_contact_loads_n','effective_pad_depth_m'])
def test_native_startup_rejects_negative_force_or_depth(key):
    data=packet();data[key].fill_(-1.)
    with pytest.raises(ValueError):mode.validate_startup_measurements(data,2)


def test_native_startup_rejects_explicit_bad_environment():
    data=packet();data['effective_pad_bad_env'][1]=True
    with pytest.raises(ValueError):mode.validate_startup_measurements(data,2)
