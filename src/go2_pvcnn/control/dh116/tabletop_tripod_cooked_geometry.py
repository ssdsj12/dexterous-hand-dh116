"""Read-only stored cooked-convex representation capture, NOT native attestation.

The caller owns an already-running, uniquely owned Isaac instance. This module
never acquires a PhysX interface, cooks, force-loads, steps, creates or sets USD
attributes. Inject only the existing get_nb_convex_mesh_data/get_convex_mesh_data
interface. CPU fixtures can use the same adapter and NEVER become native proof.

The getter exposes stored cooking data. It does not expose a PxShape handle or
prove that this representation is attached to a particular live actor. Actual
contacts, source geometry, stored cooked geometry and live-shape attachment are
separate evidence. No control, PPO, pad gate or success criterion is changed.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re

import numpy as np

SCHEMA = 'dh116-tripod-stored-cooked-geometry-v1'
SIDES = ('left', 'right')
COOK_SCHEMAS = {
    'PhysxConvexHullCollisionAPI': ('physxConvexHullCollision:hullVertexLimit',
                                  'physxConvexHullCollision:minThickness'),
    'PhysxConvexDecompositionCollisionAPI': ('physxConvexDecompositionCollision:hullVertexLimit',
        'physxConvexDecompositionCollision:maxConvexHulls','physxConvexDecompositionCollision:minThickness',
        'physxConvexDecompositionCollision:voxelResolution','physxConvexDecompositionCollision:errorPercentage',
        'physxConvexDecompositionCollision:shrinkWrap'),
    'PhysxCollisionAPI': ('physxCollision:contactOffset','physxCollision:restOffset'),
}


def _plain(value):
    """Owned JSON-compatible copy; retain nonfinite/unknown raw values as tags."""
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return {'nonfinite':str(value)}
    if value is None or type(value) in (str,float,int,bool): return value
    try: return [_plain(v) for v in value]  # carb Float3/4, Gf vectors
    except TypeError: return {'unsupported_type':type(value).__module__+'.'+type(value).__qualname__}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _under(path, root):
    return isinstance(path,str) and (path == root or path.startswith(root+'/'))


def _common_source_asset_root(paths):
    """Find the absolute asset root shared by all source collision paths.

    The kinematic FK root is often ``.../base_link`` while finger links are
    siblings below the hand asset root. Keep those two identities separate.
    """
    values=[PurePosixPath(path).parts for path in paths]
    if not values or any(not parts or parts[0]!='/' for parts in values):
        raise ValueError('absolute source collision paths required')
    common=[]
    for parts in zip(*values):
        if all(part==parts[0] for part in parts): common.append(parts[0])
        else: break
    if len(common)<2: raise ValueError('source collision paths need a common asset root')
    return str(PurePosixPath(*common))


def _matrix(value):
    a=np.asarray(value,dtype=float)
    if (a.shape!=(4,4) or not np.isfinite(a).all()
        or not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0)
        or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6,rtol=0)
        or not np.isclose(np.linalg.det(a[:3,:3]),1.,atol=1e-6,rtol=0)):
        raise ValueError('finite rigid actual body-link matrix required')
    return a.copy()


def prepare_cooked_bindings(source_models, actual_hand_roots, body_names):
    """Map all 17 source meshes/hand explicitly; no USD or SDK calls.

    Models are SourceHandGeometry objects. Body IDs are actual SDK body-name
    indices, not inferred finger indices. The caller supplies per-env actual
    hand roots and measured poses at capture time.
    """
    if not all(len(v)==2 for v in (source_models,actual_hand_roots,body_names)):
        raise ValueError('two source models/actual hand roots/body-name arrays required')
    rows=[]
    for hi,side in enumerate(SIDES):
        spec=source_models[hi].spec; source=spec['source_manifest']; kinematic_root=spec['root']
        actual=actual_hand_roots[hi]; names=list(body_names[hi])
        if (spec['side']!=side or source.get('verified') is not True
            or source.get('source_collision_count')!=17 or len(source['collision_meshes'])!=17
            or len(names)!=len(set(names)) or not isinstance(actual,str) or not actual.startswith('/')):
            raise ValueError('verified complete independent source and actual body names required')
        asset_root=_common_source_asset_root([path for item in source['collision_meshes']
            for path in (item['body_path'],item['mesh_path'],item['collision_owner'])])
        if not _under(kinematic_root,asset_root):
            raise ValueError('source kinematic root is outside source asset root')
        for item in source['collision_meshes']:
            if any(not _under(item[key],asset_root) for key in ('body_path','mesh_path','collision_owner')):
                raise ValueError('source collision ownership is outside source root')
            name=item['body_path'].rsplit('/',1)[-1]
            if names.count(name)!=1: raise ValueError('source body lacks unique actual SDK body ID')
            mapped=lambda key:actual+item[key][len(asset_root):]
            rows.append(dict(hand_index=hi,side=side,body_id=names.index(name),hand_root=actual,
                source_root=asset_root,source_kinematic_root=kinematic_root,
                source_body_path=item['body_path'],actual_body_path=mapped('body_path'),
                actual_collision_owner_path=mapped('collision_owner'),actual_mesh_path=mapped('mesh_path'),
                source_manifest=deepcopy(source),source_collision=deepcopy(item)))
    if len(set((r['hand_index'],r['body_id']) for r in rows))!=34:
        raise ValueError('34 unique actual hand/body bindings required')
    return rows


def _cook_metadata(prim):
    from pxr import Usd
    applied=list(prim.GetAppliedSchemas()); rows=[]
    registry=Usd.SchemaRegistry()
    for schema,names in COOK_SCHEMAS.items():
        definition=registry.FindAppliedAPIPrimDefinition(schema)
        for name in names:
            attr=prim.GetAttribute(name)
            fallback=definition.GetAttributeFallbackValue(name) if definition else None
            value=attr.Get() if attr else None
            rows.append(dict(name=name,api_schema=schema,api_applied=schema in applied,
                authored=bool(attr and attr.HasAuthoredValueOpinion()),
                resolved_value=_plain(value),resolved_value_known=value is not None,
                resolution_source=str(attr.GetResolveInfo().GetSource()) if attr else 'attribute_unavailable',
                schema_fallback=_plain(fallback),schema_fallback_known=fallback is not None,
                fallback_is_not_measured_runtime_setting=True))
    # Preserve any additional authored/registered cook settings; never invent defaults.
    known={r['name'] for r in rows}
    for attr in prim.GetAttributes():
        if attr.GetName() not in known and any(k in attr.GetName().lower() for k in ('convex','collision')):
            rows.append(dict(name=attr.GetName(),authored=attr.HasAuthoredValueOpinion(),
                resolved_value=_plain(attr.Get()),resolution_source=str(attr.GetResolveInfo().GetSource())))
    return dict(prim_path=str(prim.GetPath()),applied_schemas=applied,attributes=rows)


def _validate_hull(raw):
    """Validate finite topology/closed convex surface; no contact distance gate."""
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        return _validate_hull_checked(raw)


def _validate_hull_checked(raw):
    result=dict(closed_convex=False,reasons=[])
    try:
        if not isinstance(raw,dict): raise ValueError('mesh data dictionary required')
        nv,np_=raw['num_vertices'],raw['num_polygons']
        if type(nv) is not int or type(np_) is not int or nv<4 or np_<4:
            raise ValueError('integer nondegenerate vertex/polygon counts required')
        v=np.asarray(raw['vertices'],dtype=float); polygons=raw['polygons']; indices=raw['indices']
        if (v.shape!=(nv,3) or not np.isfinite(v).all() or len(polygons)!=np_
            or len(np.unique(v,axis=0))!=nv or any(type(i) is not int for i in indices)):
            raise ValueError('finite unique vertices/counts/integer indices required')
        # Numeric float32 topology residual only, NOT a physical collision tolerance.
        tol=64*np.finfo(np.float32).eps*max(float(np.max(np.abs(v))),float(np.max(np.ptp(v,axis=0))),1e-3)
        result['numeric_topology_tolerance_native_units']=tol
        used=set(); slots=set(); directed={}; volume=0.; center=v.mean(axis=0)
        for polygon in polygons:
            start,count=polygon['index_base'],polygon['num_vertices']
            if (type(start) is not int or type(count) is not int or count<3 or start<0
                or start+count>len(indices) or slots.intersection(range(start,start+count))):
                raise ValueError('valid nonoverlapping polygon index ranges required')
            slots.update(range(start,start+count)); ids=indices[start:start+count]
            if len(set(ids))!=count or min(ids)<0 or max(ids)>=nv:
                raise ValueError('polygon vertex indices invalid')
            used.update(ids); plane=np.asarray(polygon['plane'],dtype=float)
            if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-12:
                raise ValueError('finite nonzero polygon plane required')
            plane=plane/np.linalg.norm(plane[:3]); residual=v@plane[:3]+plane[3]
            if max(abs(residual[ids]))>tol or max(residual)>tol:
                raise ValueError('polygon plane is inconsistent with convex vertex enclosure')
            normal_area=np.zeros(3)
            for j in range(count):
                a,b=ids[j],ids[(j+1)%count]
                directed[(a,b)]=directed.get((a,b),0)+1
                normal_area+=np.cross(v[a]-v[ids[0]],v[b]-v[ids[0]])
            if not np.isfinite(normal_area).all() or np.dot(normal_area,plane[:3])<=tol*tol:
                raise ValueError('polygon is degenerate or winding opposes outward plane')
            for j in range(1,count-1):
                a,b,c=v[ids[0]]-center,v[ids[j]]-center,v[ids[j+1]]-center
                volume+=np.dot(a,np.cross(b,c))/6
        if (not math.isfinite(float(tol)) or not math.isfinite(float(volume))
            or slots!=set(range(len(indices))) or used!=set(range(nv))
            or any(n!=1 or directed.get((b,a))!=1 for (a,b),n in directed.items())
            or nv-len(directed)//2+np_!=2 or volume<=tol**3):
            raise ValueError('closed orientable convex 2-manifold and positive volume required')
        result.update(closed_convex=True,volume_native_units_cubed=float(volume),
            bounds_native_units=[v.min(axis=0).tolist(),v.max(axis=0).tolist()])
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError) as error:
        result['reasons'].append(str(error))
    return result


def _preflight(stage,bindings,snapshot):
    from pxr import Usd,UsdGeom,UsdPhysics
    if len(bindings)!=34 or any(sum(r['hand_index']==h for r in bindings)!=17 for h in (0,1)):
        raise ValueError('COLLIDER_BINDINGS_REQUIRE_34_TWO_HANDS')
    for key in ('actual_body_path','actual_collision_owner_path','actual_mesh_path'):
        if len({r[key] for r in bindings})!=34: raise ValueError('COLLIDER_BINDING_DUPLICATE_'+key)
    if len({(r['hand_index'],r['body_id']) for r in bindings})!=34:
        raise ValueError('COLLIDER_BODY_ID_DUPLICATE')
    for key in ('env_id','episode_id','tensor_world_physics_steps','sdk_step_index','control_step','control_substep'):
        if type(snapshot[key]) is not int or snapshot[key]<0: raise ValueError('CLOCK_INTEGER_REQUIRED_'+key)
    if snapshot['control_substep']>8: raise ValueError('CLOCK_SUBSTEP_INVALID')
    if type(snapshot['simulation_time_s']) not in (float,int): raise ValueError('CLOCK_TIME_NUMBER_REQUIRED')
    time=float(snapshot['simulation_time_s'])
    if not math.isfinite(time) or time<0: raise ValueError('CLOCK_TIME_INVALID')
    if not isinstance(snapshot.get('provenance'),dict): raise ValueError('CLOCK_PROVENANCE_REQUIRED')
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.: raise ValueError('METRE_STAGE_REQUIRED')
    roots={r['hand_root'] for r in bindings}
    if len(roots)!=2: raise ValueError('TWO_ACTUAL_HAND_ROOTS_REQUIRED')
    observed=set()
    for root in roots:
        prim=stage.GetPrimAtPath(root)
        if not prim: raise ValueError('ACTUAL_HAND_ROOT_MISSING')
        for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                enabled=UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
                if type(enabled) is not bool: raise ValueError('COLLISION_ENABLED_UNKNOWN')
                if enabled: observed.add(str(p.GetPath()))
    if observed!={r['actual_collision_owner_path'] for r in bindings}:
        raise ValueError('COLLIDER_SET_DOES_NOT_EQUAL_34_ENABLED_BOUND_OWNERS')
    for r in bindings:
        hi=r['hand_index'];side=SIDES[hi]
        if (type(hi) is not int or hi not in (0,1) or type(r['body_id']) is not int or r['body_id']<0
            or r['side']!=side or not _under(r['actual_body_path'],r['hand_root'])
            or not _under(r['actual_collision_owner_path'],r['actual_body_path'])
            or not _under(r['actual_mesh_path'],r['actual_collision_owner_path'])):
            raise ValueError('COLLIDER_SOURCE_ACTUAL_OWNERSHIP_INVALID')
        if (re.findall(r'(?:^|/)env_(\d+)(?=/|$)',r['hand_root'])!=[str(snapshot['env_id'])]
            or not r['hand_root'].endswith('/'+side+'Hand')):
            raise ValueError('CLOCK_ENV_ID_DOES_NOT_MATCH_ACTUAL_HAND_ROOT')
        source=r['source_collision'];root=r['source_root']
        for key,actual in (('body_path','actual_body_path'),('collision_owner','actual_collision_owner_path'),('mesh_path','actual_mesh_path')):
            if not _under(source[key],root) or r[actual]!=r['hand_root']+source[key][len(root):]:
                raise ValueError('COLLIDER_SOURCE_REMAP_INVALID')
        _matrix(snapshot['source_body_poses_w'][side][r['source_body_path']])


def collect_cooked_geometry(stage, cooking_interface, bindings, snapshot, *, simulation_running):
    """Capture all 34 bound colliders of ONE env. Unknown data is censored.

    Snapshot matrix convention: column vectors, world_from_body, SI metres;
    source_body_poses_w maps left/right then original source body path to 4x4.
    All clock fields in prepare integration message are mandatory. The caller
    authenticates that these values came from one actual SDK readback clock.
    Never claim that an SDK-shaped fixture or caller provenance is attestation.
    """
    result=dict(schema=SCHEMA,complete=False,censored=True,reasons=[],colliders=[],
        capture_clock=_plain({k:v for k,v in snapshot.items() if k!='source_body_poses_w'}),
        native_attested=False,live_actor_shape_attachment_proven=False,
        evidence_scope='stored_cooked_representation_getter_not_live_PxShape_support_attachment',
        simulation_running_confirmed=type(simulation_running) is bool and simulation_running,
        physics_modified=False,source_geometry_is_not_cooked_geometry=True)
    if type(simulation_running) is not bool or not simulation_running:
        result['reasons'].append('SIMULATION_RUNNING_NOT_CONFIRMED');return result
    try:
        _preflight(stage,bindings,snapshot)
    except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
        result['reasons'].append(str(error));return result
    from pxr import Usd,UsdGeom,UsdPhysics
    from .usd_surface import read_surface,layer_pins
    cache=UsdGeom.XformCache(); sources={}
    # Used solely to verify the reader does not author scene data, not to step USD.
    layers={layer.identifier:layer.ExportToString() for layer in stage.GetLayerStack()}
    for binding in bindings:
        row=dict(binding=_plain(binding),capture_clock=deepcopy(result['capture_clock']),
            complete=False,reasons=[],cooked_hulls=[],native_meshdata_status='not_queried',
            cooked_query_attempts=[],selected_query_path=None,selected_query_role=None)
        result['colliders'].append(row)
        try:
            manifest=binding['source_manifest'];path=manifest['source_path']
            if manifest.get('verified') is not True or _sha(path)!=manifest['source_sha256']:
                raise ValueError('SOURCE_ASSET_SHA_CHANGED_OR_UNVERIFIED')
            if path not in sources:
                source_stage=Usd.Stage.Open(path)
                if source_stage is None or layer_pins(source_stage)!=manifest['layer_sha256']:
                    raise ValueError('SOURCE_LAYERS_CHANGED')
                sources[path]=source_stage
            source=read_surface(sources[path],binding['source_body_path'])
            declared=binding['source_collision']
            for key in ('body_path','mesh_path','collision_owner','geometry_sha256','collision_approximation'):
                if source[key]!=declared[key]: raise ValueError('SOURCE_COLLIDER_IDENTITY_CHANGED_'+key)
            live=read_surface(stage,binding['actual_body_path'])
            if (live['mesh_path']!=binding['actual_mesh_path'] or live['collision_owner']!=binding['actual_collision_owner_path']
                or live['geometry_sha256']!=source['geometry_sha256']
                or live['collision_approximation']!=source['collision_approximation']):
                raise ValueError('SOURCE_ACTUAL_MESH_OR_OWNER_MISMATCH')
            if (not np.allclose(live['mesh_to_body'],source['mesh_to_body'],atol=1e-8,rtol=0)
                or not np.allclose(source['mesh_to_body'],declared['mesh_to_body'],atol=1e-8,rtol=0)):
                raise ValueError('SOURCE_ACTUAL_MESH_TO_BODY_TRANSFORM_MISMATCH')
            owner=stage.GetPrimAtPath(binding['actual_collision_owner_path'])
            body=stage.GetPrimAtPath(binding['actual_body_path'])
            mesh=stage.GetPrimAtPath(binding['actual_mesh_path'])
            approximation=live['collision_approximation']
            row['source_geometry']=dict(original_mesh_points=source['points_mesh'],
                original_triangles=source['triangles'],original_mesh_to_body=source['mesh_to_body'],
                source_geometry_sha256=source['geometry_sha256'],live_mesh_geometry_sha256=live['geometry_sha256'],
                live_mesh_to_body=live['mesh_to_body'],approximation=approximation,
                original_source_asset_sha256=manifest['source_sha256'])
            row['cooking_metadata']=[_cook_metadata(owner),_cook_metadata(mesh)]
            pose=_matrix(snapshot['source_body_poses_w'][binding['side']][binding['source_body_path']])
            transform=pose@np.asarray(live['mesh_to_body'])
            row['frames']=dict(convention='column_vectors_world_from_body_SI_metres',
                actual_body_link_pose_w=pose.tolist(),mesh_to_body=live['mesh_to_body'],
                usd_world_from_mesh_crosscheck_only=np.asarray(cache.GetLocalToWorldTransform(mesh)).T.tolist(),
                world_vertices_mapping='mesh_local_getter_hypothesis_from_installed_SDK_demo_not_live_shape_attachment_proof')
            if approximation not in ('convexHull','convexDecomposition'):
                raise ValueError('ACTUAL_APPROXIMATION_NOT_SUPPORTED_BY_CONVEX_GETTER')
            selected=None
            # Query the exact CollisionAPI/Xform owner first. Only a zero,
            # invalid, or unavailable owner count permits the separate mesh
            # path fallback; never substitute source paths or recook.
            for role,query in (('actual_collision_owner',binding['actual_collision_owner_path']),
                               ('actual_mesh',binding['actual_mesh_path'])):
                if role=='actual_mesh' and selected is not None: break
                attempt=dict(path=query,role=role,count=None,error=None)
                try:
                    count=cooking_interface.get_nb_convex_mesh_data(query)
                    attempt['count']=_plain(count)
                except Exception as error:
                    attempt['error']=type(error).__name__+': '+str(error);count=None
                row['cooked_query_attempts'].append(attempt)
                if type(count) is int and count>0:
                    if approximation=='convexHull' and count!=1:
                        raise ValueError('COOKED_CONVEX_COUNT_INVALID_FOR_ACTUAL_APPROXIMATION')
                    if approximation=='convexDecomposition' and count<1:
                        raise ValueError('COOKED_CONVEX_COUNT_INVALID_FOR_ACTUAL_APPROXIMATION')
                    selected=(query,role,count);break
            if selected is None:
                raise ValueError('COOKED_CONVEX_COUNT_ZERO_OR_INVALID_BOTH_EXACT_PATHS')
            query,role,count=selected
            row['queried_prim_path']=query;row['selected_query_path']=query;row['selected_query_role']=role
            row['reported_convex_count']=_plain(count)
            row['native_meshdata_status']='stored_cooking_data_returned_not_shape_attachment_attested'
            for index in range(count):
                data=cooking_interface.get_convex_mesh_data(query,index)
                validation=_validate_hull(data);raw=_plain(data)
                hull=dict(convex_index=index,raw=raw,validation=validation,
                    raw_payload_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest())
                row['cooked_hulls'].append(hull)
                if validation['closed_convex']:
                    vertices=np.asarray(data['vertices'],dtype=float)
                    with np.errstate(over='raise',invalid='raise'):
                        world_vertices=vertices@transform[:3,:3].T+transform[:3,3]
                    if not np.isfinite(world_vertices).all():raise ValueError('DERIVED_WORLD_VERTICES_NONFINITE')
                    hull['world_vertices_from_actual_body_pose']=world_vertices.tolist()
                else:row['reasons'].append('COOKED_HULL_TOPOLOGY_INVALID_'+str(index))
            row['complete']=not row['reasons']
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,AttributeError,OSError,FloatingPointError) as error:
            row['reasons'].append(type(error).__name__+': '+str(error))
    for path,source_stage in sources.items():
        expected=next(r['source_manifest'] for r in bindings if r['source_manifest']['source_path']==path)
        if layer_pins(source_stage)!=expected['layer_sha256']:
            result['reasons'].append('SOURCE_LAYERS_CHANGED_DURING_READBACK')
    if layers!={layer.identifier:layer.ExportToString() for layer in stage.GetLayerStack()}:
        result['reasons'].append('ACTUAL_USD_CHANGED_DURING_READBACK')
    if any(not row['complete'] for row in result['colliders']):result['reasons'].append('ONE_OR_MORE_COLLIDERS_CENSORED')
    result['complete']=len(result['colliders'])==34 and not result['reasons']
    result['censored']=not result['complete']
    return result
