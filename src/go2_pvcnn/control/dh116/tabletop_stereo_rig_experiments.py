"""Independent physical stereo-rig roll experiments, without image rotation.

The CPU direction/framing diagnostics establish geometry only. They do not
establish photometric texture, stereo coverage or native depth accuracy.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_contract import ROOT
from .tabletop_stereo_framing import assess_framing, project_world

VERSION='dh116-stereo-physical-rig-roll-experiment-v1'
PARENT_PROFILE=ROOT/'configs/dh116_training/stereo_camera_front_640x480_v3.json'


def roll_calibration(calibration,roll_degrees=20.):
    """Roll right/down about forward and physically move the baseline with it."""
    if type(calibration) is not StereoCalibration or not calibration.rectified:
        raise ValueError('parallel rectified StereoCalibration required')
    if type(roll_degrees) not in (int,float) or not math.isfinite(roll_degrees):
        raise ValueError('finite physical rig roll in degrees required')
    theta=math.radians(roll_degrees)
    original=calibration.T_world_left[:3,:3]
    right=math.cos(theta)*original[:,0]+math.sin(theta)*original[:,1]
    down=-math.sin(theta)*original[:,0]+math.cos(theta)*original[:,1]
    rotation=torch.stack((right,down,original[:,2]),dim=1)
    center=(calibration.T_world_left[:3,3]+calibration.T_world_right[:3,3])/2
    transforms=[]
    for sign in (-1.,1.):
        transform=torch.eye(4,dtype=torch.float64)
        transform[:3,:3]=rotation
        transform[:3,3]=center+sign*calibration.baseline_m/2*right
        transforms.append(transform)
    values=calibration.to_dict()
    values.update(T_world_left=transforms[0],T_world_right=transforms[1])
    return StereoCalibration.from_dict(values)


def roll_experiment_profile(*,block_size=5,parent_profile=PARENT_PROFILE):
    """A=physical roll/block5; B=same physical roll with only block9 changed."""
    if type(block_size) is not int or block_size not in (5,9):
        raise ValueError('the isolated experiment pair uses block_size 5 or 9')
    path=Path(parent_profile)
    parent=json.loads(path.read_text(encoding='utf-8'))
    before=StereoCalibration.from_dict(parent)
    if parent.get('profile_version')!='dh116-stereo-camera-front-v3' or (before.width,before.height)!=(640,480):
        raise ValueError('actual front-v3 640x480 parent profile required')
    after=roll_calibration(before,20.)
    framing=assess_framing(after,parent['framing_reference_world_m'],workspace_margin_m=.05)
    if not framing['all_in_view']:
        raise ValueError('rolled physical rig clips the declared actual workspace')
    summary={key:value for key,value in framing.items() if key not in ('left','right')}
    wire=torch.tensor(parent['framing_reference_world_m'][-2:],dtype=torch.float64)
    projection=project_world(after,wire)
    angles={}
    for eye in ('left','right'):
        direction=projection[eye]['uv_px'][1]-projection[eye]['uv_px'][0]
        angles[eye+'_wire_angle_degrees']=math.degrees(math.atan2(float(direction[1]),float(direction[0])))
    values=deepcopy(parent)
    values.update(after.to_dict())
    values.update(calibration_sha256=after.sha256,profile_version=VERSION,
        profile_name=f'front_physical_roll20_640x480_block{block_size}',
        enabled_by_default=False,default_accuracy_profile=False,intended_accuracy_profile=False,
        resolution_mode='physical_camera_rig_roll_and_rotated_baseline;not_image_rotation_or_resize',
        rig_roll_degrees=20.,rig_roll_axis_world=before.T_world_left[:3,2].tolist(),
        optical_right_world=after.T_world_left[:3,0].tolist(),framing_projection=summary,
        epipolar_diagnostic=dict(**angles,source_scope='CPU_projection_arithmetic',
            meaning='wire_image_direction_relative_to_horizontal_epipolar_axis',
            native_accuracy_improvement_proven=False,photometric_texture_proven=False),
        native_qualification='pending_independent_native_roll_rig_capture;CPU_geometry_does_not_establish_accuracy')
    values['depth_estimator']['block_size']=block_size
    source=Path(__file__).resolve()
    try:parent_source_path=path.resolve().relative_to(ROOT).as_posix()
    except ValueError:parent_source_path=str(path.resolve())
    values['source_metadata']=dict(source_scope='authored_physical_rig_observability_experiment',
        source_artifact=dict(path='src/go2_pvcnn/control/dh116/'+source.name,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        parent_profile_artifact=dict(path=parent_source_path,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        parent_calibration_sha256=before.sha256,parent_source_metadata=deepcopy(parent['source_metadata']),
        physical_roll_convention='positive Rodrigues about opticalForward;right=cos*r+sin*d;down=-sin*r+cos*d',
        baseline_rotates_physically=True,image_rotation_used=False,
        comparison_to_parent='block5 isolates rig roll;block9 additionally changes matching window',
        comparison_with_other_window='only depth_estimator.block_size differs;calibration and all other matcher settings identical',
        native_capture_performed_for_this_rig=False,old_rig_failed_evidence_preserved=True)
    return values
