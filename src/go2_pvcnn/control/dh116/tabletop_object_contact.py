"""Versioned object-filtered rigid-link loads, in Newtons (never net force)."""
import re
import numpy as np

GROUPS=('thumb','index','middle','ring','little','palm')

def link_groups(links):
    links=tuple(links)
    if not links or len(set(links))!=len(links):
        raise ValueError('unique deterministic collision rigid links required')
    result=[]
    for name in links:
        match=re.fullmatch(r'finger([1-5])[0-9]+_link',name)
        if match: result.append(int(match[1])-1)
        elif name in ('palm_link','base_link','hand_base_link'):result.append(5)
        else:raise ValueError(f'unmapped collision rigid link: {name}')
    return np.asarray(result)

def aggregate_object_contacts(vectors,links):
    groups=link_groups(links)
    vectors=np.asarray(vectors,dtype=float)
    if vectors.ndim not in (3,4) or vectors.shape[-3:]!=(2,len(groups),3) or not np.isfinite(vectors).all():
        raise ValueError('finite object-filtered vectors (...,2,L,3) required')
    loads=np.linalg.norm(vectors,axis=-1)
    grouped=np.stack([loads[...,groups==i].sum(axis=-1) for i in range(6)],axis=-1)
    return dict(object_link_loads=loads,grouped_object_loads=grouped,hand_object_loads=loads.sum(axis=-1))

def validate_object_contacts(measurement,links,*,batch_size=None,return_validity=False):
    """Validate structure globally, optionally report numerical validity per row.

    Missing/shape errors describe an incompatible measurement contract. Numeric
    errors describe a bad physical state and may terminate just that episode.
    """
    groups=link_groups(links)
    prefix=() if batch_size is None else (batch_size,)
    arrays={}
    valid=np.ones(1 if batch_size is None else batch_size,dtype=bool)
    for key,shape in (('object_link_loads',(2,len(groups))),('grouped_object_loads',(2,6)),('hand_object_loads',(2,))):
        try:value=np.asarray(measurement[key],dtype=float)
        except (KeyError,TypeError,ValueError) as exc:raise ValueError(f'missing/invalid {key}') from exc
        if value.shape!=prefix+shape:
            raise ValueError(f'{key} with shape {prefix+shape} required')
        valid &= (np.isfinite(value)&(value>=0)).reshape(len(valid),-1).all(axis=1)
        arrays[key]=value
    loads=arrays['object_link_loads']
    with np.errstate(invalid='ignore',over='ignore'):
        grouped=np.stack([loads[...,groups==i].sum(axis=-1) for i in range(6)],axis=-1)
        for actual,expected in ((arrays['grouped_object_loads'],grouped),
                                (arrays['hand_object_loads'],loads.sum(axis=-1))):
            valid &= np.isclose(actual,expected,rtol=1e-6,atol=1e-9).reshape(len(valid),-1).all(axis=1)
    if return_validity:
        return arrays,valid
    if not valid.all():
        raise ValueError('finite nonnegative consistent object contact loads required')
    return arrays

def object_contact_contract(links):
    groups=link_groups(links)
    return dict(version='dh116-full-hand-object-contact-v1',links=list(links),
        link_groups=groups.tolist(),groups=list(GROUPS),hands=['left','right'],units='N',
        aggregation='magnitude_per_link_then_sum_without_cross_link_vector_cancellation',
        filter='same_environment_Wire_only',
        observation_tail='left_groups_then_right_groups',
        measurement='sum_of_object_filtered_normal_force_magnitudes_per_rigid_link',
        observation_version='dh116-free-grasp-v1',observation_size=148,
        grouped_observation_scale_n=10.,pad_channels='unchanged_four_pad_diagnostics')


def clear_object_contact_slots(owner,indices):
    """Clear only reset slots; called after state readback without sensor updates."""
    for key in ('object_link_vectors','object_link_loads','grouped_object_loads','hand_object_loads'):
        getattr(owner,key)[indices]=0.
