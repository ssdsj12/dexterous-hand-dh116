"""Refine an ambiguous distance bound without changing contact permissions."""
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from go2_pvcnn.control.dh116.pad_region_guard import PadRegionGuard


def test_local_patch_sampling_refines_a_safe_small_pad_but_rejects_real_nonfront_contact():
    mesh=o3d.geometry.TriangleMesh.create_sphere(.005,resolution=30).translate([0,0,-.005])
    guard=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),[0,0,0],[0,0,1])
    nodes=np.array([[0.,-.04,.0072],[0.,.04,.0072]])
    result=guard.check(nodes,np.eye(3),np.zeros(3),.008)
    assert result['passed']
    assert result['patch_refinement_levels']==1
    assert result['patch_sampling_allowance_m']<result['sample_allowance_m']
    assert result['local_patch']['min_nonfront_clearance_m']>=.001
    nodes[:,2]=.0071
    invalid=guard.check(nodes,np.eye(3),np.zeros(3),.008)
    assert not invalid['passed']
    assert 'non-front wire clearance below 1 mm' in invalid['failures']


def test_nonfront_refinement_sharpens_bound_without_waiving_actual_penetration():
    mesh=o3d.geometry.TriangleMesh.create_sphere(.01,resolution=30).translate([0,0,-.01])
    guard=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),[0,0,0],[0,0,1])
    nodes=np.array([[0.,-.04,.0041],[0.,.04,.0041]])
    result=guard.check(nodes,np.eye(3),np.zeros(3),.008)
    assert result['nonfront_lower_bound_m']>=.001
    assert result['nonfront_sampling_allowance_m']<result['sample_allowance_m']
    assert not result['passed']  # Actual pad penetration is still rejected.
    assert 'pad penetration exceeds 1 mm' in result['failures']
    nodes[:,2]=.0039
    bad=guard.check(nodes,np.eye(3),np.zeros(3),.008)
    assert bad['nonfront_lower_bound_m']<.001


def test_recorded_loaded_pad_keeps_original_margin_with_a_tighter_bound():
    import hashlib
    import pytest
    from pxr import Usd
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
    root=Path(__file__).parents[1]
    evidence=root/'runs/dh116_contact_mpc_close_gpu_20261021_v8/summary.json'
    if not evidence.exists():pytest.skip('local physical failure evidence absent')
    summary=json.loads(evidence.read_text())
    path=root/'configs/contact_mpc_close_v1/calibrated/contact/manifest.json'
    record=load_mesh_manifest(path,hashlib.sha256(path.read_bytes()).hexdigest())['left_thumb']
    refs=json.loads((path.parent.parent/'point_references.json').read_text())
    ref=next(r for r in refs['references'] if r['side']=='left' and r['digit']=='thumb')
    mesh=o3d.io.read_triangle_mesh(str(record['resolved_path']));offset=np.array(record['body_local_mesh_com_m'])
    guard=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.array(ref['point'])-offset,ref['normal'])
    model=SideKinematics(Usd.Stage.Open(str(root/'assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd')),'left')
    pose=model.fk(summary['measured_source_guard_active_q']['left'])[0][model.bodies[ref['body_name']]]
    old=summary['measured_source_guard_failure']['pad_regions']['left_thumb']
    nodes=np.asarray(summary['last_attempted_record']['rod_state']['rod_position']).T
    result=guard.check(nodes,pose[:3,:3],pose[:3,3]+pose[:3,:3]@offset,old['wire_radius_m'])
    assert result['passed'],result
    assert result['nonfront_lower_bound_m']>.00115
    assert guard.partition_sha256==old['partition_sha256']  # Same protected surface.


def test_recorded_index_patch_refines_sampling_without_changing_the_allowed_patch():
    import hashlib
    import pytest
    from pxr import Usd
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
    root=Path(__file__).parents[1]
    evidence=root/'runs/dh116_contact_mpc_close_gpu_20261021_v10/summary.json'
    if not evidence.exists():pytest.skip('local physical failure evidence absent')
    summary=json.loads(evidence.read_text())
    path=root/'configs/contact_mpc_close_v1/calibrated/contact/manifest.json'
    record=load_mesh_manifest(path,hashlib.sha256(path.read_bytes()).hexdigest())['left_index']
    refs=json.loads((path.parent.parent/'point_references.json').read_text())
    ref=next(r for r in refs['references'] if r['side']=='left' and r['digit']=='index')
    mesh=o3d.io.read_triangle_mesh(str(record['resolved_path']));offset=np.array(record['body_local_mesh_com_m'])
    guard=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.array(ref['point'])-offset,ref['normal'])
    model=SideKinematics(Usd.Stage.Open(str(root/'assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd')),'left')
    pose=model.fk(summary['measured_source_guard_active_q']['left'])[0][model.bodies[ref['body_name']]]
    old=summary['measured_source_guard_failure']['pad_regions']['left_index']
    nodes=np.asarray(summary['last_attempted_record']['rod_state']['rod_position']).T
    result=guard.check(nodes,pose[:3,:3],pose[:3,3]+pose[:3,:3]@offset,old['wire_radius_m'])
    assert result['passed'],result
    assert result['local_patch']['min_nonfront_clearance_m']>.00115
    assert result['patch_sampling_allowance_m']<result['sample_allowance_m']
    assert result['patch_refinement_levels']>0
    assert guard.partition_sha256==old['partition_sha256']
