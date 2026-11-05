import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_physics_cloning')


class Cloner:
    def __init__(self):
        self.calls=[];self.output=object()
    def clone(self,**kwargs):
        self.calls.append(('clone',kwargs));return self.output
    def filter_collisions(self,*args,**kwargs):
        self.calls.append(('filter_collisions',args,kwargs))


def test_replicate_preserves_exact_clone_flags_and_output_identity(module):
    cloner=Cloner();paths=['/World/envs/env_0','/World/envs/env_1']
    original=paths.copy()
    result=module.clone_physics_environments(cloner,object(),paths)
    assert result is cloner.output
    assert cloner.calls==[('clone',dict(source_prim_path=paths[0],prim_paths=paths,
        replicate_physics=True,copy_from_source=False,enable_env_ids=True))]
    assert cloner.calls[0][1]['prim_paths'] is paths
    assert paths==original


def test_usd_parse_preserves_geometry_inheritance_and_adds_explicit_collision_groups(module):
    cloner=Cloner();paths=['/World/envs/env_0','/World/envs/env_1'];original=paths.copy()
    context_calls=[]
    def get_context():
        context_calls.append(True);return SimpleNamespace(prim_path='/World/physicsScene')
    result=module.clone_physics_environments(cloner,SimpleNamespace(get_physics_context=get_context),paths,'usd_parse')
    assert result is cloner.output and context_calls==[True]
    assert cloner.calls==[
        ('clone',dict(source_prim_path=paths[0],prim_paths=paths,
            replicate_physics=False,copy_from_source=False,enable_env_ids=False)),
        ('filter_collisions',('/World/physicsScene','/World/collisions',paths),{'global_paths':[]})]
    assert cloner.calls[0][1]['prim_paths'] is paths
    assert cloner.calls[1][1][2] is paths
    assert paths==original


@pytest.mark.parametrize('mode,paths',[('unknown',['/env_0']),('', ['/env_0']),('replicate',[]),('usd_parse',[])])
def test_invalid_requests_fail_before_any_native_call(module,mode,paths):
    cloner=Cloner()
    with pytest.raises(ValueError):
        module.clone_physics_environments(cloner,object(),paths,mode)
    assert cloner.calls==[]
