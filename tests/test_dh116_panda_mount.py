import pytest


def test_real_dual_asset_structure_and_no_overwrite(tmp_path):
    pytest.importorskip('pxr.Usd')
    from scripts.build_fixed_dual_panda_dh116 import build
    from pxr import Usd, UsdPhysics, UsdGeom
    import numpy as np
    output = tmp_path/'mounted'
    report = build(output)
    assert report['physical_dof'] == 36
    assert report['active_dof'] == 26
    stage = Usd.Stage.Open(str(output/'fixed_dual_panda_dh116.usd'))
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]) == 2
    cache = UsdGeom.XformCache()
    for side in ('left','right'):
        arm = f'/FixedDualPandaDH116/{side}_arm'
        assert not stage.GetPrimAtPath(arm+'/dh116/root_joint').IsActive()
        wrist = stage.GetPrimAtPath(arm+f'/{side}_panda_link8')
        palm = stage.GetPrimAtPath(arm+'/dh116/base_link')
        np.testing.assert_allclose(np.array(cache.GetLocalToWorldTransform(wrist)),
                                   np.array(cache.GetLocalToWorldTransform(palm)),atol=1e-7)
        mount = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f'/FixedDualPandaDH116/joints/{side}_mount'))
        assert list(map(str,mount.GetBody1Rel().GetTargets())) == [str(palm.GetPath())]
    assert all('o6' not in str(p.GetPath()).lower() for p in stage.Traverse())
    with pytest.raises(ValueError, match='overwrite'):
        build(output)
