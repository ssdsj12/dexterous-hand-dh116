"""Explicit simulation candidates, not manufacturer actuator calibration."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class LoadParameters:
    max_force_n: float = 1.
    force_rate_n_s: float | None = .5  # None: apply the requested target immediately.
    deadband_n: float = .02
    stiffness_max: float = 20.
    stiffness_rate: float = 10.
    effort_limit_nm: float = .2
    calibration_status: str = 'simulation_estimate'

    def __post_init__(self):
        bounds = dict(max_force_n=(.1,32.), force_rate_n_s=(0.,10.),
            deadband_n=(0.,.5), stiffness_max=(10.,200.), stiffness_rate=(0.,200.),
            effort_limit_nm=(0.,2.))
        for key,(low,high) in bounds.items():
            value = getattr(self,key)
            if key == 'force_rate_n_s' and value is None:
                continue
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f'{key} must be finite within ({low},{high}]')
        if self.calibration_status != 'simulation_estimate':
            raise ValueError('this profile cannot assert hardware calibration')

    def as_dict(self):
        return asdict(self)


LEGACY_LOAD_PARAMETERS = LoadParameters()


def read_load_parameters(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != 'dh116-simulation-load-v1':
        raise ValueError('unknown simulation load profile')
    return LoadParameters(**payload['parameters'])


def verify_drive_readback(stiffness, damping, effort, parameters=None):
    """Fail before training if the simulator did not apply the requested limits."""
    parameters = parameters or LEGACY_LOAD_PARAMETERS
    stiffness,damping,effort = [np.asarray(x,dtype=float) for x in (stiffness,damping,effort)]
    if (stiffness.ndim != 2 or stiffness.shape[1] != 6
            or damping.shape != stiffness.shape or effort.shape != stiffness.shape
            or not all(np.isfinite(x).all() for x in (stiffness,damping,effort))):
        raise ValueError('finite N x 6 active drive readback required')
    if np.any(stiffness < 10.-1e-5) or np.any(stiffness > parameters.stiffness_max+1e-5):
        raise ValueError('simulator stiffness outside selected profile')
    if not np.allclose(damping,.2,atol=1e-6,rtol=0):
        raise ValueError('simulator damping differs from selected profile')
    if not np.allclose(effort,parameters.effort_limit_nm,atol=1e-6,rtol=0):
        raise ValueError('simulator effort limit differs from selected profile')


def _verify_velocity_readback(velocity, names, instances, passive_limit):
    from .linear_coupling import ACTIVE, JOINTS
    from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S, passive_velocity_limit
    passive_limit = passive_velocity_limit('adaptive_grasp', passive_limit)
    from .tabletop_palm_mount import JOINT_NAMES,CONTRACT as WRIST
    valid=(set(names)==set(JOINTS) or set(names)==set(JOINTS)|set(JOINT_NAMES))
    if len(names)!=len(set(names)) or not valid:
        raise ValueError('velocity readback requires eleven DH116 joints and optional complete wrist carriage')
    if (instances < 1 or velocity.shape != (instances, len(names))
            or not np.isfinite(velocity).all()):
        raise ValueError('finite all-instance N x 11 velocity limit readback required')
    expected = np.asarray([WRIST['velocity_limit_m_s'] if name in JOINT_NAMES else
                           (ACTIVE_VELOCITY_LIMIT_RAD_S if name in ACTIVE else passive_limit) for name in names])
    if not np.allclose(velocity, expected[None, :], atol=1e-6, rtol=0):
        raise ValueError('simulator velocity limits differ from selected active/passive limits')


def physics_load_readback(hands, parameters=None, *, passive_velocity_limit_rad_s=None):
    """Actual PhysX views; compact samples plus all-instance drive validation."""
    from .linear_coupling import ACTIVE
    result=[]
    for hand in hands:
        view=hand.root_physx_view
        read=lambda method: getattr(view,method)().detach().cpu().numpy()
        stiffness,damping,effort=[read(method) for method in
            ('get_dof_stiffnesses','get_dof_dampings','get_dof_max_forces')]
        velocity=read('get_dof_max_velocities')
        if passive_velocity_limit_rad_s is not None:
            _verify_velocity_readback(velocity,hand.joint_names,stiffness.shape[0],
                                      passive_velocity_limit_rad_s)
        ids=[hand.joint_names.index(n) for n in ACTIVE]
        verify_drive_readback(stiffness[:,ids],damping[:,ids],effort[:,ids],parameters)
        from .tabletop_palm_mount import JOINT_NAMES,CONTRACT as WRIST
        if set(JOINT_NAMES).issubset(hand.joint_names):
            wrist_ids=[hand.joint_names.index(n) for n in JOINT_NAMES]
            for values,expected in [(stiffness,WRIST['stiffness_n_m']),(damping,WRIST['damping_n_s_m']),(effort,WRIST['effort_limit_n'])]:
                if not np.allclose(values[:,wrist_ids],expected,rtol=0,atol=1e-5):
                    raise ValueError('wrist carriage drive readback mismatch')
        mass,inertia,com=[read(method) for method in ('get_masses','get_inertias','get_coms')]
        result.append(dict(instances=int(stiffness.shape[0]),joint_names=hand.joint_names,
            body_names=hand.body_names,stiffness_nm_per_rad=stiffness[:1].tolist(),
            damping_nms_per_rad=damping[:1].tolist(),effort_limit_nm=effort[:1].tolist(),
            joint_limits_rad=read('get_dof_limits')[:1].tolist(),
            velocity_limits_rad_s=velocity[:1].tolist(),
            sample_body_masses_kg=mass[:1].tolist(),sample_body_inertias_kg_m2=inertia[:1].tolist(),
            sample_body_com_poses=com[:1].tolist(),total_mass_kg_range=[float(mass.sum(axis=1).min()),float(mass.sum(axis=1).max())],
            all_instance_drive_limits_verified=True))
        if set(JOINT_NAMES).issubset(hand.joint_names):
            result[-1]['wrist_carriage']=WRIST
            result[-1]['joint_units']={n:('linear_m_N' if n in JOINT_NAMES else 'angular_rad_Nm') for n in hand.joint_names}
        if passive_velocity_limit_rad_s is not None:
            result[-1]['all_instance_velocity_limits_verified']=True
    return dict(hands=result,hardware_calibration_complete=False,
        inertial_model='unchanged manufacturer CAD; finger32 distribution anomaly unresolved',
        coupling_model='source URDF affine approximation; not nonlinear MJCF linkage',
        torque_diagnostic='implicit PD estimate, not measured solver drive torque',
        contact_diagnostic='object-filtered normal forces on pads, not total contact wrench')
