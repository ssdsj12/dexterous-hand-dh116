"""Initialization-only SDK-clock adapter for an unchanged protected collector.

Only the absent TensorWorld clock field changes. All original 34-owner/source/
geometry/cooking/read-only checks remain in the cloned functions. No native
interface is acquired here; fixture/caller provenance is never attestation.
"""
import ast
import hashlib
import math
from pathlib import Path

ORIGINAL_SHA256='ddf703d192958b2a1a88634c8a8433f46d43e43c15575cd02cee0a5803a75219'
SCHEMA='dh116-teacher-sdk-clock-stored-cooked-v1'


def validate_sdk_clock(snapshot):
    """Strictly distinguish real SDK tick2 from an unavailable TensorWorld."""
    if (snapshot.get('tensor_world_physics_steps','missing') is not None
        or snapshot.get('clock_source')!='isaac_sim_context_without_tensor_world'
        or not isinstance(snapshot.get('provenance'),dict)
        or snapshot['provenance'].get('tensor_world_owner')!='not_instantiated'):
        raise ValueError('SDK_CLOCK_ABSENT_TENSOR_WORLD_SCOPE_REQUIRED')
    for name,expected in (('sdk_step_index',2),('env_id',0),('control_step',0),('control_substep',0)):
        if type(snapshot.get(name)) is not int or snapshot[name]!=expected:
            raise ValueError('SDK_CLOCK_INITIALIZATION_IDENTITY_INVALID_'+name)
    time=snapshot.get('simulation_time_s')
    if type(time) not in (int,float) or not math.isfinite(time) or not math.isclose(time,2/480,abs_tol=2e-6,rel_tol=0):
        raise ValueError('SDK_CLOCK_INITIALIZATION_TIME_INVALID')


def build_sdk_collector(*,source_path=None):
    """Clone two exact SHA-bound functions into owned globals; never patch original."""
    from . import tabletop_tripod_cooked_geometry as original
    path=Path(source_path) if source_path is not None else Path(original.__file__)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('protected collector SHA differs')
    tree=ast.parse(raw.decode('utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in ('_preflight','collect_cooked_geometry')]
    if [n.name for n in nodes]!=['_preflight','collect_cooked_geometry']:
        raise ValueError('protected collector function layout differs')
    preflight=nodes[0];changes=0
    expected=('env_id','episode_id','tensor_world_physics_steps','sdk_step_index','control_step','control_substep')
    for node in ast.walk(preflight):
        if isinstance(node,ast.Tuple) and all(isinstance(v,ast.Constant) for v in node.elts):
            if tuple(v.value for v in node.elts)==expected:
                node.elts=[v for v in node.elts if v.value!='tensor_world_physics_steps'];changes+=1
    if changes!=1:raise ValueError('exact one original clock-key tuple required')
    preflight.body.insert(0,ast.parse('validate_sdk_clock(snapshot)').body[0])
    namespace=original.__dict__.copy()
    namespace.update(SCHEMA=SCHEMA,validate_sdk_clock=validate_sdk_clock,
                     __name__=__name__+'.sha_bound_clone')
    compiled=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(compiled,str(path)+'[sdk-clock-initialization-only]','exec'),namespace)
    return namespace['collect_cooked_geometry']


def collect_sdk_cooked_geometry(stage,cooking_interface,bindings,snapshot,*,simulation_running):
    return build_sdk_collector()(stage,cooking_interface,bindings,snapshot,
                                 simulation_running=simulation_running)
