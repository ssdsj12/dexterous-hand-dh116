from pathlib import Path
import pytest
from pxr import Usd,UsdGeom
from tools.relocate_assets import relocate


def test_nested_reference_is_local(tmp_path):
    source=tmp_path/'old';source.mkdir()
    child=Usd.Stage.CreateNew(str(source/'child.usda'))
    child.SetDefaultPrim(UsdGeom.Xform.Define(child,'/Child').GetPrim());child.GetRootLayer().Save()
    parent=Usd.Stage.CreateNew(str(source/'parent.usda'))
    parent.DefinePrim('/Root').GetReferences().AddReference(str(source/'child.usda'))
    parent.GetRootLayer().Save()
    output=tmp_path/'new folder'
    result=relocate(source/'parent.usda',output)
    stage=Usd.Stage.Open(str(output/'parent.usda'))
    assert stage.GetPrimAtPath('/Root').IsValid()
    assert all(Path(l.realPath).is_relative_to(output) for l in stage.GetUsedLayers() if l.realPath)
    assert len(result['layers'])==2


def test_unresolved_reference_rejected(tmp_path):
    source=tmp_path/'bad.usda';stage=Usd.Stage.CreateNew(str(source))
    stage.DefinePrim('/Root').GetReferences().AddReference('missing.usda');stage.GetRootLayer().Save()
    with pytest.raises(ValueError):relocate(source,tmp_path/'out')
