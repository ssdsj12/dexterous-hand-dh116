"""Current-model gravity feedforward; explicit simulation effort bounds only."""
import math
import torch
from .linear_coupling import JOINTS,ACTIVE,coupling_matrix


def compensated_efforts(gravity,joint_names,side,arm_limit,hand_limit):
    if side not in ('left','right'):raise ValueError('invalid side')
    names=tuple(joint_names);arms=tuple(f'{side}_panda_joint{i}' for i in range(1,8))
    if len(names)!=18 or set(names)!=set(arms+JOINTS):raise ValueError('invalid joint names')
    if (not isinstance(gravity,torch.Tensor) or gravity.ndim!=2 or gravity.shape[1]!=18
        or not gravity.is_floating_point() or not torch.isfinite(gravity).all()):raise ValueError('invalid gravity tensor')
    if any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in (arm_limit,hand_limit)):
        raise ValueError('limits must be positive finite simulation values')
    arm_ids=[names.index(n) for n in arms];hand_ids=[names.index(n) for n in JOINTS]
    active_ids=[names.index(n) for n in ACTIVE]
    C=torch.as_tensor(coupling_matrix(),device=gravity.device,dtype=gravity.dtype)
    hand=gravity[:,hand_ids]@C
    if not torch.isfinite(hand).all():raise ValueError('gravity folding overflow')
    if (gravity[:,arm_ids].abs()>arm_limit).any() or (hand.abs()>hand_limit).any():
        failures=[]
        for values,group,limit in ((gravity[:,arm_ids],arms,arm_limit),(hand,ACTIVE,hand_limit)):
            for row,column in torch.nonzero(values.abs()>limit).cpu().tolist():
                failures.append(f'{group[column]} demand={float(values[row,column]):.9g} limit={limit:.9g} Nm')
        raise ValueError('gravity compensation exceeds explicit simulation effort limit: '+side+' '+', '.join(failures))
    command=torch.zeros_like(gravity)
    command[:,arm_ids]=gravity[:,arm_ids];command[:,active_ids]=hand
    return command


def drive_headroom(feedforward,joint_names,side,arm_limit,hand_limit):
    """Keep |implicit PD| + |separate feedforward| inside one motor budget.

    PhysX drive max force alone is not a bound on the sum with externally
    submitted joint effort. Passive joints retain exactly zero drive budget.
    """
    # Reuse the strict name/shape/limit validation; input is already folded.
    validated=compensated_efforts(feedforward,joint_names,side,arm_limit,hand_limit)
    if not torch.equal(validated,feedforward):raise ValueError('feedforward must already be active-only')
    caps=torch.zeros_like(feedforward)
    for i,name in enumerate(joint_names):
        if name in ACTIVE:caps[:,i]=hand_limit
        elif name.startswith(side+'_panda_joint'):caps[:,i]=arm_limit
    remaining=caps-feedforward.abs()
    if (remaining<0).any():raise ValueError('no drive headroom')
    return remaining
