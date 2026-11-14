"""Physical-rig observability geometry only; no native accuracy assertion."""
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_framing import assess_framing, project_world

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/dh116_training/stereo_camera_front_640x480_v3.json'


def module():
    name='go2_pvcnn.control.dh116.tabletop_stereo_rig_experiments'
    assert importlib.util.find_spec(name),'independent physical rig experiment helper missing'
    return importlib.import_module(name)


def test_optical_roll_moves_both_physical_cameras_and_preserves_parallel_calibrated_stereo():
    before=StereoCalibration.from_json(PARENT);after=module().roll_calibration(before,20.)
    theta=math.radians(20);r=before.T_world_left[:3,0];d=before.T_world_left[:3,1]
    right=math.cos(theta)*r+math.sin(theta)*d
    down=-math.sin(theta)*r+math.cos(theta)*d
    center=(before.T_world_left[:3,3]+before.T_world_right[:3,3])/2
    torch.testing.assert_close(after.K_left,before.K_left,rtol=0,atol=0)
    torch.testing.assert_close(after.K_right,before.K_right,rtol=0,atol=0)
    assert (after.width,after.height,after.baseline_m)==(640,480,.08)
    for name,offset in [('left',-.04),('right',.04)]:
        transform=getattr(after,'T_world_'+name)
        torch.testing.assert_close(transform[:3,0],right)
        torch.testing.assert_close(transform[:3,1],down)
        torch.testing.assert_close(transform[:3,2],before.T_world_left[:3,2],rtol=0,atol=0)
        torch.testing.assert_close(transform[:3,3],center+offset*right)
        assert not torch.equal(transform[:3,3],getattr(before,'T_world_'+name)[:3,3])
        torch.testing.assert_close(transform[:3,:3].T@transform[:3,:3],torch.eye(3,dtype=torch.float64))
        assert float(torch.det(transform[:3,:3]))==pytest.approx(1.)
    relative=torch.linalg.inv(after.T_world_left)@after.T_world_right
    torch.testing.assert_close(relative[:3,:3],torch.eye(3,dtype=torch.float64))
    torch.testing.assert_close(relative[:3,3],torch.tensor([.08,0.,0.],dtype=torch.float64))
    assert after.sha256!=before.sha256


def test_wire_image_direction_changes_relative_to_epipolar_without_rolling_an_image():
    before=StereoCalibration.from_json(PARENT);after=module().roll_calibration(before,20.)
    endpoints=torch.tensor([[.43,-.25,.368],[.43,.25,.368]],dtype=torch.float64)
    for eye in ('left','right'):
        old=project_world(before,endpoints)[eye]['uv_px'];new=project_world(after,endpoints)[eye]['uv_px']
        old_direction=old[1]-old[0];direction=new[1]-new[0]
        assert float(old_direction[1])==pytest.approx(0.,abs=1e-10)
        assert float(direction[1]/direction[0])==pytest.approx(-math.tan(math.radians(20)),abs=1e-12)
        assert abs(float(direction[1]))>50.
    data=json.loads(PARENT.read_text())
    framing=assess_framing(after,data['framing_reference_world_m'],workspace_margin_m=.05)
    assert framing['expanded_point_count']==48 and framing['all_in_view'] is True
    assert framing['minimum_border_margin_px']>1
    assert framing['native_visibility_qualified'] is False


@pytest.mark.parametrize('block_size',[5,9])
def test_experiment_profile_binds_sources_and_changes_only_rig_plus_declared_window(block_size):
    m=module();path=ROOT/f'configs/dh116_training/stereo_camera_front_roll20_640x480_block{block_size}_experiment_v1.json'
    assert path.is_file(),'explicit roll experiment profile missing'
    value=json.loads(path.read_text());parent=json.loads(PARENT.read_text())
    c=StereoCalibration.from_dict(value)
    assert c.sha256==m.roll_calibration(StereoCalibration.from_dict(parent),20.).sha256
    expected=dict(parent['depth_estimator'],block_size=block_size)
    assert value['depth_estimator']==expected
    for key in ('center_world_m','target_world_m','horizontal_fov_degrees','baseline_m','K_left','K_right',
                'disparity_range_px','depth_range_m','resource_limits','framing_reference_world_m'):
        assert value[key]==parent[key]
    source=value['source_metadata']
    assert source['source_artifact']['sha256']==hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    assert source['parent_profile_artifact']['sha256']==hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert source['parent_profile_artifact']['path']=='configs/dh116_training/stereo_camera_front_640x480_v3.json'
    assert source['parent_calibration_sha256']==parent['calibration_sha256']
    assert source['native_capture_performed_for_this_rig'] is False
    assert value['enabled_by_default'] is False and value['default_accuracy_profile'] is False
    assert value['rig_roll_degrees']==20. and value['framing_projection']['all_in_view'] is True
    assert value['epipolar_diagnostic']['left_wire_angle_degrees']==pytest.approx(-20.)
    assert value['epipolar_diagnostic']['right_wire_angle_degrees']==pytest.approx(-20.)


def test_window_pair_shares_one_new_calibration_and_v3_source_is_unchanged():
    m=module();a=m.roll_experiment_profile(block_size=5);b=m.roll_experiment_profile(block_size=9)
    assert a['calibration_sha256']==b['calibration_sha256']!=StereoCalibration.from_json(PARENT).sha256
    aa=dict(a['depth_estimator']);bb=dict(b['depth_estimator']);aa.pop('block_size');bb.pop('block_size')
    assert aa==bb
    assert hashlib.sha256((ROOT/'src/go2_pvcnn/control/dh116/tabletop_stereo_framing.py').read_bytes()).hexdigest()==(
        '884487259d8f69f215431d7899d457dfb18e505bfc8f903336a5217acc673550')
