"""Independent validation-only optical-Z reference; never an Actor input.

Native CameraView 'depth' binds distance_to_image_plane in audited Isaac 4.5
source. This reader never renders or advances physics. Mismatched render-product
clocks reject the pair; metadata is never relabelled to make it align.
"""
import copy
import importlib
import hashlib
import inspect
import json
import math
from pathlib import Path
import re

import torch

AUDITED_CAMERA_VIEW_SHA256='58557bbcb900574a4b9858c13e9e479b96fd5dfe09d872b07d5c010cca64d99e'
AUDITED_INSTANCE_ANNOTATOR_SHA256='756dbfb4f262e623606bcbff2ddb315be723cf371e7a72983493d7e9977ddd56'


def validate_reference_timing(pair,reference,*,native=False):
    """Require the same actual render clocks, not only equal physics time."""
    left=pair.get('render_timing');right=reference.get('render_timing')
    if not native and left is None and right is None:return
    if not isinstance(left,dict) or not isinstance(right,dict):raise ValueError('independent render clocks unavailable')
    keys=('frame_id','simulation_time','reference_time_numerator','reference_time_denominator',
          'render_frame_number','swh_frame_number')
    if any(left.get(key)!=right.get(key) for key in keys):raise ValueError('independent render product clock mismatch')
    if native:
        from .tabletop_vision_camera import paired_render_timestamp
        for timing in (left,right):
            actual=paired_render_timestamp(dict(referenceTimeNumerator=timing.get('reference_time_numerator'),
                referenceTimeDenominator=timing.get('reference_time_denominator')),timing.get('isaac_read_times'),
                legacy_time=timing.get('legacy_simulation_time'))
            if any(actual[key]!=timing.get(key) for key in keys):raise ValueError('raw render clock does not reproduce timing metadata')


def resolve_wire_instance_mask(ids,info,wire_prim_paths,*,path_validator=None,
                               wire_render_prim_paths=None,reference_valid=None):
    """Resolve actual instance-ID prim paths, never substring/class/colour labels.

    Unknown metadata is an explicit unavailable reference. Native callers supply
    an actual USD stage validator; CPU/offline callers can validate archived path
    readback. Each environment is matched only against its own real Wire root.
    """
    mapping={};background=set();validated=set();renderer_namespaces={};ignored=[];unresolved=[]
    def result(mask=None,reason=None,per_env=None,complete=False):
        return dict(mask=mask,id_to_path=mapping,per_env_wire_ids=per_env or [],
            stage_validated_paths=sorted(validated),blocked_reasons=[] if reason is None else [reason],
            background_ids=sorted(background),non_wire_renderer_instance_namespaces=renderer_namespaces,
            unresolved_ids=unresolved,unresolved_ids_outside_reference_validity=ignored,wire_inventory_complete=complete)
    failed=lambda reason:result(reason=reason)
    if not isinstance(ids,torch.Tensor) or ids.ndim!=3 or ids.dtype not in (torch.int32,torch.int64):
        return failed('instance_ids_must_be_signed_integer_BHW')
    if (not isinstance(wire_prim_paths,(list,tuple)) or len(wire_prim_paths)!=len(ids)
            or len(set(wire_prim_paths))!=len(wire_prim_paths)):
        return failed('actual_per_environment_wire_roots_unavailable_or_duplicated')
    pattern=r'/(?:[A-Za-z_][A-Za-z0-9_:]*)(?:/[A-Za-z_][A-Za-z0-9_:]*)*'
    def valid_path(value):return isinstance(value,str) and re.fullmatch(pattern,value) is not None
    if any(not valid_path(root) for root in wire_prim_paths):return failed('actual_wire_root_is_not_absolute_USD_prim_path')
    if path_validator is not None and any(not path_validator(root) for root in wire_prim_paths):
        return failed('actual_wire_root_missing_from_USD_stage_readback')
    validated.update(wire_prim_paths)
    inventory=wire_render_prim_paths
    if inventory is not None:
        if (not isinstance(inventory,list) or len(inventory)!=len(ids)
                or any(not isinstance(paths,list) or not paths for paths in inventory)):
            return failed('complete_per_environment_actual_Wire_render_geometry_inventory_required')
        for root,paths in zip(wire_prim_paths,inventory):
            if (len(set(paths))!=len(paths) or any(not valid_path(path) or not (path==root or path.startswith(root+'/'))
                    or (path_validator is not None and not path_validator(path)) for path in paths)):
                return failed('Wire_render_geometry_inventory_not_actual_stage_descendants')
            validated.update(paths)
        if (not isinstance(reference_valid,torch.Tensor) or reference_valid.shape!=ids.shape
                or reference_valid.dtype!=torch.bool or reference_valid.device!=ids.device):
            return failed('actual_optical_Z_reference_validity_required_for_instance_inventory')
    labels=info.get('idToLabels') if isinstance(info,dict) else None
    if isinstance(labels,str):
        try:labels=json.loads(labels)
        except ValueError:return failed('unsupported_idToLabels_JSON')
    if not isinstance(labels,dict):return failed('unsupported_instance_info_format_missing_idToLabels')
    for key,value in labels.items():
        if isinstance(key,bool) or not re.fullmatch('[0-9]+',str(key)):return failed('unsupported_instance_ID_key')
        number=int(key)
        if isinstance(value,str):
            if value=='BACKGROUND' and number==0:background.add(number);continue
            if value.lstrip().startswith('{'):
                try:value=json.loads(value)
                except ValueError:continue
        path=value.get('primPath') if isinstance(value,dict) else value
        if not valid_path(path):
            # The audited fast annotator returns renderer instance-token paths.
            # An explicit non-Wire renderer namespace is not an unknown ID.
            # With complete actual Wire inventory, it may be excluded without
            # pretending that its synthetic token is a real USD prim. Never
            # infer a Wire ID or strip a suffix to certify Wire geometry.
            match=re.fullmatch(r'(.+)\.proto_mesh_id[0-9]+',path) if isinstance(path,str) else None
            base=match[1] if match is not None else None
            if (inventory is None or not valid_path(base)
                    or any(base==root or base.startswith(root+'/') for root in wire_prim_paths)):
                continue
            renderer_namespaces[number]=base
            continue
        if path_validator is not None and not path_validator(path):continue
        mapping[number]=path;validated.add(path)
    seen=set(torch.unique(ids).detach().cpu().tolist())
    unresolved=sorted(value for value in seen if value not in mapping and value not in background and value not in renderer_namespaces)
    for value in unresolved:
        if inventory is None or bool(((ids==value)&reference_valid).any()):
            return failed('visible_instance_ID_has_no_unambiguous_stage_validated_prim_path')
        ignored.append(value)  # Explicitly unresolved, NOT guessed background.
    per_env=[];mask=torch.zeros_like(ids,dtype=torch.bool)
    for env,root in enumerate(wire_prim_paths):
        owned=[key for key,path in mapping.items() if path==root or path.startswith(root+'/')]
        if not owned:return failed('Wire_prim_not_resolved_to_instance_ID_for_environment_'+str(env))
        if inventory is not None and any(not any(path==mapping[key] or path.startswith(mapping[key]+'/')
                for key in owned) for path in inventory[env]):
            return failed('actual_Wire_render_geometry_missing_from_instance_label_map_for_environment_'+str(env))
        per_env.append(sorted(owned))
        mask[env]=torch.isin(ids[env],torch.tensor(owned,device=ids.device,dtype=ids.dtype))
    return result(mask=mask,per_env=per_env,complete=inventory is not None)


class _NativeValidationDepthReader:
    def __init__(self,camera,wire_prim_paths=None):
        from .tabletop_stereo_camera import _NativeRGBPairReader
        from .tabletop_vision_camera import _CameraAxesDeviceMixin, numeric_slot_order
        rgb=camera.reader
        if not isinstance(rgb,_NativeRGBPairReader):raise ValueError('validation reference requires an actual native RGB rig')
        base=importlib.import_module('isaacsim.sensors.camera').CameraView
        source=Path(inspect.getfile(base)).resolve();source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if source_sha!=AUDITED_CAMERA_VIEW_SHA256:
            raise ValueError('unaudited CameraView source; optical-Z annotator mapping requires explicit source review')
        self.sdk_source_artifact=dict(path=str(source),sha256=source_sha)
        self.wire_prim_paths=list(wire_prim_paths) if wire_prim_paths is not None else None
        class DeviceSafeDepthView(_CameraAxesDeviceMixin,base):pass
        rep=importlib.import_module('omni.replicator.core')
        annotator=Path(inspect.getfile(importlib.import_module('omni.replicator.core.scripts.annotators_default'))).resolve()
        annotator_sha=hashlib.sha256(annotator.read_bytes()).hexdigest()
        if annotator_sha!=AUDITED_INSTANCE_ANNOTATOR_SHA256:
            raise ValueError('unaudited renderer instance annotator source')
        self.sdk_instance_source_artifact=dict(path=str(annotator),sha256=annotator_sha)
        for name in ('sim','calibration','device','origins','n','count','Usd','UsdGeom','stage'):
            setattr(self,name,getattr(rgb,name))
        self.core=rgb.core
        c=self.calibration
        self.view=DeviceSafeDepthView(prim_paths_expr='/World/DH116StereoRGB/camera_.*',name='dh116_stereo_validation_depth',
            camera_resolution=(c.width,c.height),output_annotators=['depth','instance_id_segmentation_fast'],reset_xform_properties=False)
        self.paths=list(self.view.prim_paths);self.order=numeric_slot_order(self.paths,self.count).to(self.device)
        width=int(self.view.tiled_resolution[0]);self.columns=width//c.width
        if width!=self.columns*c.width:raise ValueError('invalid validation reference tile layout')
        self.reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime',device='cpu')
        self.times=rep.AnnotatorRegistry.get_annotator('IsaacReadTimes',device='cpu')
        self.reference.attach([self.view.get_render_product_path()]);self.times.attach([self.view.get_render_product_path()])

    def _wire_render_geometry(self):
        if self.wire_prim_paths is None:return None
        result=[]
        for path in self.wire_prim_paths:
            root=self.stage.GetPrimAtPath(path)
            if not root.IsValid():return None
            # Actual USD renderable geometry, not names/seed/GT object pixels.
            result.append(sorted(str(prim.GetPath()) for prim in self.Usd.PrimRange(root)
                if prim.IsA(self.UsdGeom.Gprim) and self.UsdGeom.Imageable(prim).ComputeVisibility()!=self.UsdGeom.Tokens.invisible))
        return result

    def read(self):
        from .tabletop_stereo_camera import _NativeRGBPairReader
        from .tabletop_vision_camera import paired_render_timestamp, detile
        reference=self.reference.get_data()
        if not isinstance(reference,dict) or 'referenceTimeNumerator' not in reference:
            raise ValueError('validation reference render frame unavailable')
        legacy=float(self.core.get_sim_time_at_swh_frame(int(reference['referenceTimeNumerator'])))
        before=paired_render_timestamp(reference,self.times.get_data(),legacy_time=legacy)
        c=self.calibration
        depth=detile(self.view.get_depth_tiled(device='cuda'),count=self.count,width=c.width,height=c.height,
            columns=self.columns,channels=1).to(self.device)[self.order].reshape(self.n,2,c.height,c.width)
        segmentation=None;segmentation_info=None;segmentation_error=None
        try:
            value=self.view.get_data('instance_id_segmentation_fast',tiled=True)
            if not isinstance(value,tuple) or len(value)!=2:raise ValueError('unsupported segmentation get_data return format')
            data,segmentation_info=value
            if isinstance(data,torch.Tensor):data=data.detach().cpu()
            elif hasattr(data,'numpy'):data=torch.as_tensor(data.numpy())
            else:data=torch.as_tensor(data)
            # wp.uint32 -> signed long BEFORE indexing/unique/isin operations.
            segmentation=detile(data.long(),count=self.count,width=c.width,height=c.height,
                columns=self.columns,channels=1).to(self.device)[self.order].reshape(self.n,2,c.height,c.width)
        except (ValueError,TypeError,RuntimeError) as error:
            segmentation_error='instance_reference_unavailable:'+str(error)
        after=paired_render_timestamp(self.reference.get_data(),self.times.get_data(),legacy_time=legacy)
        if (before['frame_id'],before['simulation_time'],before['render_frame_number'])!=(after['frame_id'],after['simulation_time'],after['render_frame_number']):
            raise ValueError('validation reference frame/time changed during independent depth read')
        # Reuse only the camera-calibration readback method against THIS depth
        # CameraView; it never reads object state or RGB-D actor features.
        _NativeRGBPairReader._readback(self)
        inventory=self._wire_render_geometry()
        reference_valid=torch.isfinite(depth[:,0])&(depth[:,0]>0)
        resolution=resolve_wire_instance_mask(segmentation[:,0],segmentation_info,self.wire_prim_paths,
            path_validator=lambda path:self.stage.GetPrimAtPath(path).IsValid(),
            wire_render_prim_paths=inventory,reference_valid=reference_valid) if segmentation is not None and inventory is not None else None
        return dict(depth_left_m=depth[:,0].float(),depth_right_m=depth[:,1].float(),
            frame_id=before['frame_id'],simulation_time=before['simulation_time'],calibration_sha256=c.sha256,
            render_timing=before,sdk_camera_view_source_artifact=self.sdk_source_artifact,
            sdk_instance_annotator_source_artifact=self.sdk_instance_source_artifact,
            raw_instance_ids=None if segmentation is None else segmentation[:,0],raw_instance_info=segmentation_info,
            wire_prim_paths=self.wire_prim_paths,wire_render_prim_paths=inventory,wire_instance_resolution=resolution,
            segmentation_blocked_reason=segmentation_error)


class ValidationDepthReference:
    def __init__(self,reader):self.reader=reader

    @classmethod
    def from_camera(cls,camera,wire_prim_paths=None):return cls(_NativeValidationDepthReader(camera,wire_prim_paths))

    @torch.no_grad()
    def read(self,pair):
        value=self.reader.read()
        if not isinstance(value,dict):raise ValueError('independent reference frame unavailable')
        for name in ('frame_id','simulation_time','calibration_sha256'):
            if value.get(name)!=pair.get(name):raise ValueError('reference RGB pair frame/time/calibration mismatch: '+name)
        native=isinstance(self.reader,_NativeValidationDepthReader)
        validate_reference_timing(pair,value,native=native)
        left,right=value.get('depth_left_m'),value.get('depth_right_m')
        shape=(len(pair['left_rgb']),*pair['left_rgb'].shape[-2:])
        if not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor) or left.shape!=shape or right.shape!=shape:
            raise ValueError('reference optical depth shape mismatch')
        valid=torch.isfinite(left)&(left>0)
        from .tabletop_vision_camera import brown_mask
        candidate=brown_mask(pair['left_rgb'].permute(0,2,3,1),float_rgb_is_255=False)
        resolution=value.get('wire_instance_resolution') or {}
        return dict(version='dh116-stereo-depth-reference-v1',source_scope='native' if native else 'cpu_fixture',
            validation_only=True,reference_used_as_actor_input=False,source_units='optical_axis_z_m',
            reference_source='rendered_depth_validation_only',frame_id=value['frame_id'],
            simulation_time=value['simulation_time'],calibration_sha256=value['calibration_sha256'],
            depth_m=left.clone(),valid=valid.clone(),right_depth_m=right.clone(),
            rgb_candidate_mask=candidate.clone(),candidate_mask_source='RGB_brown_candidate_not_semantic_ground_truth',
            wire_reference_mask=resolution.get('mask'),wire_mask_source_scope=('native_resolved_instance_id' if native else 'cpu_resolved_instance_id') if resolution.get('mask') is not None else 'unresolved',
            wire_mask_blocked_reasons=resolution.get('blocked_reasons',[value.get('segmentation_blocked_reason') or 'instance_reference_unavailable']),
            raw_instance_ids=value.get('raw_instance_ids'),raw_instance_info=copy.deepcopy(value.get('raw_instance_info')),
            wire_prim_paths=copy.deepcopy(value.get('wire_prim_paths')),
            wire_render_prim_paths=copy.deepcopy(value.get('wire_render_prim_paths')),
            wire_instance_resolution=resolution,actual_wire_path_source='RigidObject.root_physx_view.prim_paths' if native else 'cpu_fixture',
            render_timing=copy.deepcopy(value.get('render_timing')),
            sdk_instance_annotator_source_artifact=copy.deepcopy(value.get('sdk_instance_annotator_source_artifact')),
            sdk_camera_view_source_artifact=copy.deepcopy(value.get('sdk_camera_view_source_artifact')))


def archive_instance_reference(reference,path):
    """Archive unmodified ID/map readback separately from the derived mask.

    Unknown formats are archived too; a stored artifact does not grant source
    qualification. The native promotion gate independently resolves this data.
    """
    if reference.get('raw_instance_ids') is None:return None
    names=('source_scope','frame_id','simulation_time','calibration_sha256','render_timing',
           'raw_instance_ids','raw_instance_info','wire_prim_paths','actual_wire_path_source',
           'sdk_camera_view_source_artifact','sdk_instance_annotator_source_artifact','wire_render_prim_paths')
    raw={key:copy.deepcopy(reference.get(key)) for key in names}
    raw.update(version='dh116-wire-instance-reference-v2',validation_only=True,reference_valid=reference['valid'].clone(),
        stage_validated_paths=copy.deepcopy(reference.get('wire_instance_resolution',{}).get('stage_validated_paths',[])))
    path=Path(path)
    with path.open('xb') as stream:torch.save(raw,stream)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def capture_validation_pair(camera,reference,identity,*,max_attempts=8):
    """Retry the WHOLE real RGB pair; reference read itself performs no render."""
    failures=[]
    for _ in range(max_attempts):
        pair=camera.sample(identity=identity)
        try:
            ref=reference.read(pair)
            return pair,ref,failures
        except ValueError as error:
            failures.append(dict(frame_id=pair.get('frame_id'),simulation_time=pair.get('simulation_time'),error=str(error)))
    raise RuntimeError('same-frame independent reference unavailable; failed pairs='+repr(failures))


def compare_reference(pair,reference):
    """Error arithmetic; visible-wire completeness uses an explicit ID mask."""
    for key in ('frame_id','simulation_time','calibration_sha256'):
        if pair.get(key)!=reference.get(key):raise ValueError('reference alignment mismatch')
    estimate,target=pair['depth_m'],reference['depth_m']
    good_ref=reference['valid']&torch.isfinite(target)&(target>0)
    if 'evaluation_depth_range_m' in reference:
        near,far=reference['evaluation_depth_range_m'];good_ref&=(target>=near)&(target<=far)
    valid=pair['valid']&torch.isfinite(estimate)&good_ref
    if estimate.shape!=target.shape or valid.shape!=target.shape:raise ValueError('depth/reference shape mismatch')
    error=(estimate[valid]-target[valid]).float()
    count=int(valid.sum());denominator=int(good_ref.sum())
    candidate=reference.get('rgb_candidate_mask')
    candidate_den=good_ref&candidate if isinstance(candidate,torch.Tensor) else torch.zeros_like(valid)
    candidate_count=int(candidate_den.sum())
    candidate_valid=valid&candidate_den
    wire=reference.get('wire_reference_mask');wire_total=wire_reconstructed=None;wire_error=None;wire_count=None
    wire_reason='no_resolved_independent_instance_reference'
    if isinstance(wire,torch.Tensor):
        if wire.shape!=valid.shape or wire.dtype!=torch.bool:raise ValueError('wire instance reference mask shape/dtype mismatch')
        if reference.get('wire_mask_source_scope') not in ('native_resolved_instance_id','cpu_resolved_instance_id'):
            raise ValueError('wire reference must derive from explicit resolved instance IDs')
        if pair.get('K_left') is None:raise ValueError('calibration K required for 3D completeness')
        from .tabletop_stereo_geometry import backproject
        truth=backproject(target,pair['K_left'],good_ref)
        reconstructed=backproject(estimate,pair['K_left'],pair['valid'])
        distance=(truth-reconstructed).norm(dim=-1)
        wire_total=int((wire&good_ref).sum())
        wire_reconstructed=int((wire&valid&(distance<=.005)).sum())
        wire_error=(estimate[wire&valid]-target[wire&valid]).float();wire_count=len(wire_error)
        wire_reason=None if wire_total else 'no_reference_visible_wire_points'
    return dict(depth_mae_m=float(error.abs().mean()) if count else None,
        depth_rmse_m=float(error.square().mean().sqrt()) if count else None,
        depth_p95_m=float(error.abs().quantile(.95)) if count else None,
        depth_validity_coverage=count/denominator if denominator else 0.,depth_valid_count=count,
        depth_reference_valid_count=denominator,
        rgb_candidate_point_cloud_completeness=float(candidate_valid.sum())/candidate_count if candidate_count else None,
        rgb_candidate_reference_points=candidate_count,rgb_candidate_reconstructed_points=int(candidate_valid.sum()),
        candidate_mask_source=reference.get('candidate_mask_source'),
        wire_point_cloud_completeness=wire_reconstructed/wire_total if wire_total else None,
        wire_reference_visible_points=wire_total,wire_reconstructed_points=wire_reconstructed,
        wire_depth_mae_m=float(wire_error.abs().mean()) if wire_count else None,
        wire_depth_rmse_m=float(wire_error.square().mean().sqrt()) if wire_count else None,
        wire_depth_p95_m=float(wire_error.abs().quantile(.95)) if wire_count else None,
        wire_depth_count=wire_count,wire_depth_validity_coverage=wire_count/wire_total if wire_total else None,
        wire_completeness_scope='dense_reference_visible_wire_pixels;not_hidden_entire_object',
        wire_completeness_reason=wire_reason)
