"""Measured physical initial conditions for a fresh close-stage experiment.

Unlike a qualified stage checkpoint, this imports no approach/grasp evidence
and does not restore opaque PhysX solver caches or phase qualification. Optional
explicit controller references preserve loading progress without importing dwell.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE, JOINTS, expand


def initial_wire_geometry(payload,nominal_nodes,*,base_radius_m=.008):
    """Screen the geometry that will actually be restored, not a straight proxy."""
    if payload is None:return np.asarray(nominal_nodes,dtype=float).copy(),float(base_radius_m)
    nodes=np.asarray(payload['rod']['dynamic']['rod_position'],dtype=float).T
    rest=np.asarray(payload['rod']['reference']['rest_lengths'],dtype=float)
    if (nodes.ndim!=2 or nodes.shape[1]!=3 or len(nodes)<2 or rest.shape!=(len(nodes)-1,)
            or not np.isfinite(nodes).all() or not np.isfinite(rest).all() or np.any(rest<=0)
            or not np.isfinite(base_radius_m) or base_radius_m<=0):
        raise ValueError('finite restored wire geometry and rest lengths required')
    lengths=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    if np.any(lengths<1e-10):raise ValueError('restored wire has collapsed elements')
    radius=base_radius_m*np.sqrt(rest/lengths)
    return nodes.copy(),float(radius.max())


def _validate_controller_reference(reference,source_step):
    depth=np.asarray(reference['commanded_penetration_m'],dtype=float)
    if (depth.shape!=(2,3) or not np.isfinite(depth).all() or np.any(depth<0.) or np.any(depth>.001)
            or type(reference.get('source_step')) is not int
            or not 0<=reference['source_step']<=source_step):
        raise ValueError('invalid saved load reference')
    if 'tracking_bias_rad' in reference:
        bias=np.asarray(reference['tracking_bias_rad'],dtype=float)
        if bias.shape!=(12,) or not np.isfinite(bias).all() or np.any(np.abs(bias)>np.pi/180+1e-8):
            raise ValueError('invalid saved hand-only tracking bias')


def from_contact_row(row, *, source_run, asset_sha256, contact_manifest_sha256, rod_physical,
                     controller_reference=None):
    if not row.get('passed') or row.get('grasp_sequence',{}).get('stage')!='close':
        raise ValueError('a measured valid close row is required')
    robot={}
    for side in ('left','right'):
        measured=row['sides'][side]
        targets=dict(zip([f'{side}_panda_joint{i}' for i in range(1,8)],row['targets'][side]['arm']))
        targets.update(zip(JOINTS,expand(np.asarray(row['targets'][side]['hand']))))
        names=measured['joint_names']
        robot[side]=dict(joint_names=names,q=measured['actual_joint_positions'],
            qd=measured['actual_joint_velocities'],target=[float(targets[n]) for n in names])
    rod=copy.deepcopy(rod_physical);rod['dynamic']=copy.deepcopy(row['rod_state'])
    payload=dict(schema='dh116-close-initialization-v1',grasp_qualified=False,
        asset_sha256=asset_sha256,contact_manifest_sha256=contact_manifest_sha256,
        source=dict(run=str(source_run),step=int(row['step'])),robot=robot,rod=rod,
        contact_loads_n=[float(row['contact']['normal_samples'][side+'_'+digit]['normal_load_n'])
            for side in ('left','right') for digit in ('thumb','index','middle')])
    if controller_reference is not None:
        _validate_controller_reference(controller_reference,int(row['step']))
        payload['controller_reference']=controller_reference
    return copy.deepcopy(payload)


def load_close_initialization(path, sha256, *, asset_sha256, contact_manifest_sha256):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha256:
        raise ValueError('close initial-state SHA mismatch')
    payload=json.loads(data)
    json.dumps(payload,allow_nan=False)
    if (payload.get('schema')!='dh116-close-initialization-v1' or payload.get('grasp_qualified') is not False
            or payload.get('asset_sha256')!=asset_sha256
            or payload.get('contact_manifest_sha256')!=contact_manifest_sha256
            or set(payload.get('robot',{}))!={'left','right'}):
        raise ValueError('close initial-state identity or diagnostic scope mismatch')
    if 'contact_loads_n' in payload:
        loads=np.asarray(payload['contact_loads_n'],dtype=float)
        if loads.shape!=(6,) or not np.isfinite(loads).all() or np.any(loads<0):
            raise ValueError('invalid initial measured contact loads')
    if 'controller_reference' in payload:
        _validate_controller_reference(payload['controller_reference'],payload['source']['step'])
    return payload


def rebase_hand_stiffness(payload, *, old_stiffness, new_stiffness):
    """Preserve initial proportional torque when increasing simulation Kp."""
    if not np.isfinite([old_stiffness,new_stiffness]).all() or not 0<old_stiffness<=new_stiffness:
        raise ValueError('positive nondecreasing simulation hand stiffness required')
    result=copy.deepcopy(payload)
    for robot in result['robot'].values():
        names=robot['joint_names'];target=np.asarray(robot['target']).copy();q=np.asarray(robot['q'])
        indices=[names.index(name) for name in ACTIVE]
        target[indices]=q[indices]+(old_stiffness/new_stiffness)*(target[indices]-q[indices])
        full=expand(target[indices])
        for name,value in zip(JOINTS,full):target[names.index(name)]=value
        robot['target']=target.tolist()
    result['hand_stiffness_nm_rad']=float(new_stiffness)
    if old_stiffness!=new_stiffness:
        result.get('controller_reference',{}).pop('tracking_bias_rad',None)
    result['initial_target_rebase']=dict(old_stiffness_nm_rad=float(old_stiffness),
        new_stiffness_nm_rad=float(new_stiffness),scope='preserve initial proportional drive torque; not hardware calibration')
    return result


def reframe_close_initialization(payload,states,*,alignment):
    """Fresh offline pregrasp alignment; never teleport an existing simulation.

    Preserve rod dynamics, joint velocities and initial active drive deflection.
    The caller must separately screen all new geometry and then requalify in
    physics. No predecessor stage, grasp, or path qualification is imported.
    """
    if set(states)!={'left','right'}:raise ValueError('both aligned sides required')
    result=copy.deepcopy(payload)
    for side,robot in result['robot'].items():
        active=np.asarray(states[side],dtype=float)
        if active.shape!=(13,) or not np.isfinite(active).all():raise ValueError('finite aligned 13D state required')
        full=dict(zip([f'{side}_panda_joint{i}' for i in range(1,8)],active[:7]))
        full.update(zip(JOINTS,expand(active[7:])))
        names=robot['joint_names'];oldq=np.asarray(robot['q']);oldtarget=np.asarray(robot['target'])
        newq=np.array([full[n] for n in names]);target=newq+(oldtarget-oldq)
        followers=expand(target[[names.index(n) for n in ACTIVE]])
        for name,value in zip(JOINTS,followers):target[names.index(name)]=value
        robot['q']=newq.tolist();robot['target']=target.tolist()
    result['pregrasp_alignment']=dict(copy.deepcopy(alignment),physics_qualified=False,motion_qualified=False)
    result['grasp_qualified']=False
    return result


def robot_initial_arrays(saved, names, limits):
    if len(names)!=18 or set(names)!=set(saved['joint_names']):
        raise ValueError('physical joint identity mismatch')
    indices=[saved['joint_names'].index(n) for n in names]
    arrays={key:np.asarray(saved[key],dtype=float)[indices].copy() for key in ('q','qd','target')}
    limits=np.asarray(limits)
    if (limits.shape!=(18,2) or any(value.shape!=(18,) or not np.isfinite(value).all() for value in arrays.values())
            or any(np.any(arrays[key]<limits[:,0]) or np.any(arrays[key]>limits[:,1]) for key in ('q','target'))):
        raise ValueError('initial physical joint state outside live limits')
    caps=np.array([.151 if '_panda_joint' in name else .31 for name in names])
    if np.any(np.abs(arrays['qd'])>caps):
        raise ValueError('initial joint velocity exceeds runtime guard')
    return arrays
