"""Runtime USD authoring tests; not physical stability evidence."""
import numpy as np
import pytest
from pxr import Usd, UsdPhysics
from go2_pvcnn.control.dh116.linear_coupling import MIMIC


def scene():
    stage=Usd.Stage.CreateInMemory()
    for follower,master,_,_ in MIMIC:
        for name in (follower,master):
            UsdPhysics.RevoluteJoint.Define(stage,'/Hand/joints/'+name).CreateAxisAttr('X')
        stage.GetPrimAtPath('/Hand/joints/'+follower).AddAppliedSchema('PhysxMimicJointAPI:rotX')
    return stage


def test_compliance_authors_all_five_couplings_in_runtime_layer():
    from go2_pvcnn.control.dh116.runtime_mimic import configure_mimic_compliance
    stage=scene()
    assert configure_mimic_compliance(stage,'/Hand',200.,1.) == 5
    for follower,_,_,_ in MIMIC:
        prim=stage.GetPrimAtPath('/Hand/joints/'+follower)
        assert prim.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Get()==200.
        assert prim.GetAttribute('physxMimicJoint:rotX:dampingRatio').Get()==1.


@pytest.mark.parametrize('frequency,damping',[(0,1),(-1,1),(np.nan,1),(200,.5),(200,np.inf)])
def test_compliance_rejects_invalid_or_underdamped_values(frequency,damping):
    from go2_pvcnn.control.dh116.runtime_mimic import configure_mimic_compliance
    with pytest.raises(ValueError): configure_mimic_compliance(scene(),'/Hand',frequency,damping)


def test_explicit_rigid_mode_authors_zero_frequency_on_all_followers():
    from go2_pvcnn.control.dh116.runtime_mimic import configure_mimic_compliance
    stage=scene()
    assert configure_mimic_compliance(stage,'/Hand',0.,1.,allow_rigid=True)==5
    for follower,_,_,_ in MIMIC:
        prim=stage.GetPrimAtPath('/Hand/joints/'+follower)
        assert prim.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Get()==0.
    with pytest.raises(ValueError):
        configure_mimic_compliance(stage,'/Hand',-1.,1.,allow_rigid=True)
