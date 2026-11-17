"""CPU startup regressions, no native RGB/physics qualification."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native


def managers(enabled=False, registered=False):
    events=[]
    state=dict(enabled=enabled, registered=registered)
    def enable(name, value):
        events.append((name, value))
        state['enabled']=value
        state['registered']=value
    manager=SimpleNamespace(
        is_extension_enabled=lambda name:state['enabled'],
        set_extension_enabled_immediate=enable)
    registry=SimpleNamespace(get_registered_annotators=lambda:
        ['IsaacReadTimes'] if state['registered'] else [])
    return manager,registry,events


def test_installed_core_nodes_is_enabled_before_clock_annotator_use():
    manager,registry,events=managers()
    native.ensure_clock_annotators(manager,registry)
    assert events==[('isaacsim.core.nodes',True)]
    assert 'IsaacReadTimes' in registry.get_registered_annotators()


def test_already_enabled_core_nodes_is_not_restarted():
    manager,registry,events=managers(True,True)
    native.ensure_clock_annotators(manager,registry)
    assert events==[]


def test_enabled_but_unregistered_clock_is_rejected_without_fabrication():
    manager,registry,events=managers(True,False)
    with pytest.raises(RuntimeError,match='IsaacReadTimes'):
        native.ensure_clock_annotators(manager,registry)
    assert events==[]


def test_clock_registration_precedes_private_rig_and_tiled_initialization():
    tree=ast.parse(Path(native.__file__).read_text())
    constructor=next(n for n in ast.walk(tree)
        if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    calls=[n for n in ast.walk(constructor) if isinstance(n,ast.Call)]
    ensure=next((n.lineno for n in calls
        if isinstance(n.func,ast.Name) and n.func.id=='ensure_clock_annotators'),None)
    rig=next(n.lineno for n in calls
        if isinstance(n.func,ast.Attribute) and n.func.attr=='define_rig')
    assert ensure is not None and ensure<rig


def test_hidden_reference_time_is_not_required_in_public_registry_list():
    manager,registry,events=managers(True,True)
    assert 'ReferenceTime' not in registry.get_registered_annotators()
    native.ensure_clock_annotators(manager,registry)
