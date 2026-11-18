from pathlib import Path
from pxr import Usd, UsdPhysics
import pytest
PhysxSchema = pytest.importorskip('pxr.PhysxSchema', reason='requires running Isaac schema plugins')
import numpy as np

def test_runtime_mimic_preserves_source_and_authors_five_relations():
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic
    from go2_pvcnn.control.dh116.linear_coupling import MIMIC
    source = Path(__file__).parents[1] / 'assets/fixed_dual_panda_dh116_v1/dependencies/d134680ab619_DH116_left.usd'
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/Hand').GetReferences().AddReference(str(source))
    assert install_hand_mimic(stage, '/Hand') == 5
    for follower, master, ratio, offset in MIMIC:
        api = PhysxSchema.PhysxMimicJointAPI(stage.GetPrimAtPath('/Hand/joints/'+follower), 'rotX')
        assert str(api.GetReferenceJointRel().GetTargets()[0]) == '/Hand/joints/'+master
        assert np.isclose(api.GetGearingAttr().Get(), -ratio)
        assert np.isclose(api.GetOffsetAttr().Get(), -np.degrees(offset))
