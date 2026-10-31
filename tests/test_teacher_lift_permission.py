"""Exploratory motion permission never replaces measured force qualification."""
import importlib.util,os
from pathlib import Path
import torch

def helper():
    path=Path(os.environ.get('DH116_TEACHER_REFERENCE',Path(__file__).parents[1]/
        'src/go2_pvcnn/control/dh116/tabletop_native_teacher_reference.py'))
    spec=importlib.util.spec_from_file_location('teacher_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_explicit_exploration_can_lift_before_force_qualification_and_after_reset_waits_again():
    ready=torch.tensor([False,False,True])
    age=torch.tensor([6.,0.,.5],dtype=torch.float64)
    assert helper().lift_motion_permission(ready,age,6.).tolist()==[True,False,True]
    assert ready.tolist()==[False,False,True]

def test_default_teacher_permission_preserves_force_readiness():
    ready=torch.tensor([False,True])
    age=torch.tensor([100.,0.],dtype=torch.float64)
    assert torch.equal(helper().lift_motion_permission(ready,age,None),ready)
