"""Read-only saved-artifact error diagnostics; never starts Isaac or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_error_diagnostics import (
    distribution, region_metrics, reference_visibility, wire_bands,
    rejection_summary, sampled_cost_curves, replay_depth_conditions,
)


def checked_artifact(binding):
    path=Path(binding['path'])
    if not path.is_file():raise FileNotFoundError('missing artifact: '+str(path)+' SHA='+binding['sha256'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('artifact SHA mismatch: '+str(path))
    return path


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_identity():
    import importlib
    import importlib.metadata
    import platform
    names=('tabletop_stereo_error_cli','tabletop_stereo_error_diagnostics','tabletop_stereo_error_geometry',
        'tabletop_stereo_depth','tabletop_stereo_calibration','tabletop_stereo_geometry','tabletop_stereo_perception',
        'tabletop_stereo_matcher_candidates','tabletop_stereo_matching_costs','tabletop_stereo_pyramid')
    sources={}
    for name in names:
        path=Path(importlib.import_module('.'+name,__package__).__file__).resolve()
        sources[name]=dict(path=str(path),sha256=_sha(path))
    versions={}
    for name in ('numpy','scipy','matplotlib','opencv-python'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(sources=sources,python=platform.python_version(),torch=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,packages=versions,device='CPU',isaac_started=False,
        config=dict(visibility_tolerance_m=.001,visibility_sensitivity_m=[.0005,.001,.002],
            correct_3d_metric_m=.005,reference_edge_diagnostic_threshold_m=.005,
            reference_used_as_actor_input=False))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def load_capture(proof_path):
    proof_path=Path(proof_path);p=json.loads(proof_path.read_text(encoding='utf-8'))
    bindings={key:checked_artifact(p[key]) for key in ('capture_artifact','calibration_artifact',
        'validation_config_artifact','performance_artifact','completion_artifact')}
    for key in ('source_artifact','config_artifact'):checked_artifact(p['producer'][key])
    completion=json.loads(bindings['completion_artifact'].read_text(encoding='utf-8'))
    produced={str(Path(x['path'])):x for x in completion['produced_artifacts']}
    pending_path=proof_path.parent/'native_stereo_proof_pending.json'
    if str(pending_path) not in produced:raise ValueError('completion does not bind pending proof')
    pending=json.loads(checked_artifact(produced[str(pending_path)]).read_text(encoding='utf-8'))
    for key in ('producer','capture_artifact','calibration_artifact','validation_config_artifact','reference_artifacts',
                'performance_artifact','cohort_sha256','sample_count','action_contract_sha256','state_contract_sha256'):
        if p.get(key)!=pending.get(key):raise ValueError('completed/pending proof identity mismatch: '+key)
    capture=torch.load(bindings['capture_artifact'],map_location='cpu',weights_only=False)
    c=StereoCalibration.from_json(bindings['calibration_artifact'])
    config=p['producer']['kwargs']['depth_estimator']
    refs=[]
    if len(capture['frames'])!=len(p['reference_artifacts']):raise ValueError('capture/reference frame count mismatch')
    for frame,binding in zip(capture['frames'],p['reference_artifacts']):
        ref=torch.load(checked_artifact(binding),map_location='cpu',weights_only=False)
        for key in ('frame_id','simulation_time','calibration_sha256'):
            if ref[key]!=frame[key]:raise ValueError('reference frame identity mismatch: '+key)
        if frame['calibration_sha256']!=c.sha256:raise ValueError('calibration identity mismatch')
        if frame.get('depth_source')!='computed_from_left_right_RGB':raise ValueError('saved prediction is not RGB-only computed depth')
        if (ref.get('source_units')!='optical_axis_z_m' or not ref.get('validation_only')
                or ref.get('reference_used_as_actor_input') is not False):
            raise ValueError('separate optical-Z validation reference required')
        wire_binding=ref.get('wire_mask_source_artifact')
        if wire_binding:checked_artifact(wire_binding)
        refs.append(ref)
    return capture,refs,c,config,p


def _label_masks(reference,frame):
    ids=reference.get('raw_instance_ids');raw_info=reference.get('raw_instance_info')
    info=raw_info.get('idToLabels') if isinstance(raw_info,dict) else None
    if not isinstance(ids,torch.Tensor) or not isinstance(info,dict) or not info:
        return dict(table=None,hand_occluder=None,palm=None,finger=None)
    masks={name:torch.zeros_like(ids,dtype=torch.bool) for name in ('table','hand_occluder','palm','finger')}
    seen=set()
    for i,row in enumerate(frame['identity']):
        prefix=f"/World/envs/env_{row['env_id']}/"
        for raw_id,label in info.items():
            path=label if isinstance(label,str) else label.get('primPath','') if isinstance(label,dict) else ''
            if not path.startswith(prefix):continue
            hit=ids[i]==int(raw_id)
            if '/Table/' in path:masks['table'][i]|=hit;seen.add('table')
            if '/leftHand/' in path or '/rightHand/' in path:
                masks['hand_occluder'][i]|=hit;seen.add('hand_occluder')
                if '/base_link/' in path:masks['palm'][i]|=hit;seen.add('palm')
                if '/finger' in path:masks['finger'][i]|=hit;seen.add('finger')
    return {name:value if name in seen else None for name,value in masks.items()}


def resolve_proof(row):
    report=checked_artifact(row['report_artifact'])
    content=json.loads(report.read_text(encoding='utf-8'))
    matches=[x for x in content['resolutions'] if (x['width'],x['height'])==(row['width'],row['height'])]
    if len(matches)!=1:raise ValueError('benchmark resolution identity ambiguous or absent')
    selected=matches[0]
    capture_report=checked_artifact(dict(path=selected['capture_report_path'],sha256=selected['capture_report_sha256']))
    cap=json.loads(capture_report.read_text(encoding='utf-8'))
    proof=Path(selected['stereo_qualification']);pending=Path(cap['stereo_qualification'])
    if (pending.name!='native_stereo_proof_pending.json' or proof.name!='native_stereo_proof_completed.json'
            or proof.parent!=pending.parent or proof.parent!=capture_report.parent):
        raise ValueError('benchmark/capture proof identity mismatch')
    if ((cap['width'],cap['height'])!=(row['width'],row['height'])
            or cap['cohort_sha256']!=row['cohort_sha256'] or selected['cohort_sha256']!=row['cohort_sha256']):
        raise ValueError('benchmark resolution/cohort identity mismatch')
    return report,proof,dict(capture_report_artifact=dict(path=str(capture_report),sha256=_sha(capture_report)),
        width=row['width'],height=row['height'],cohort_sha256=row['cohort_sha256'])


@torch.no_grad()
def analyze_frame(frame,reference,calibration,config,*,replay=None,max_cost_pixels=24):
    pred=frame['depth_m'];valid=frame['valid'];target=reference['depth_m']
    eligible=reference['valid']&torch.isfinite(target)&(target>=config['min_depth_m'])&(target<=config['max_depth_m'])
    wire=reference.get('wire_reference_mask')
    resolved=(reference.get('wire_mask_source_scope')=='native_resolved_instance_id'
        and reference.get('wire_instance_resolution',{}).get('wire_inventory_complete') is True
        and not reference.get('wire_mask_blocked_reasons'))
    wire=wire if isinstance(wire,torch.Tensor) and resolved else None
    gray=frame['left_rgb'].mean(1);r=config['block_size']//2
    mean=F.avg_pool2d(gray[:,None],config['block_size'],1,r)[:,0]
    std=(F.avg_pool2d(gray.square()[:,None],config['block_size'],1,r)[:,0]-mean.square()).clamp_min(0).sqrt()
    vis=reference_visibility(target,reference.get('right_depth_m'),eligible,calibration.K_left,calibration.baseline_m)
    regions=dict(global_image=torch.ones_like(eligible),**_label_masks(reference,frame),
        low_texture=std<config['texture_threshold'],textured=std>=config['texture_threshold'],**vis)
    if wire is not None:
        from .tabletop_stereo_perception import _thin
        import numpy as np
        thin=torch.as_tensor(np.stack([_thin(row.numpy()) for row in wire]),dtype=torch.bool)
        band=F.max_pool2d(thin.float()[:,None],3,1,1)[:,0]>0
        regions.update(wire_bands(wire));regions['wire_centerline_band']=wire&band
        regions['wire_common_visible']=None if vis['common_visible'] is None else wire&vis['common_visible']
        regions['wire_left_only_occluded']=None if vis['left_only_occluded'] is None else wire&vis['left_only_occluded']
    else:
        for key in ('wire','wire_edge','wire_interior','wire_centerline_band','wire_common_visible','wire_left_only_occluded'):
            regions[key]=None
    from .tabletop_stereo_error_geometry import wire_geometry_regions
    geometry_regions,geometry_metadata=wire_geometry_regions(wire,torch.where(eligible,target,float('nan')),calibration.K_left)
    regions.update(geometry_regions)
    # Reference edge is a diagnostic ROI. The existing matcher has no such reject filter.
    dx=F.pad((target[:,:,1:]-target[:,:,:-1]).abs(),(0,1));dy=F.pad((target[:,1:]-target[:,:-1]).abs(),(0,0,0,1))
    regions['reference_depth_edge']=eligible&((dx>.005)|(dy>.005))
    for lo,hi in ((.2,.4),(.4,.6),(.6,.8),(.8,1.2)):
        regions[f'distance_{lo:.1f}_{hi:.1f}m']=(target>=lo)&((target<hi) if hi<1.2 else (target<=hi))
    if replay is None:replay=replay_depth_conditions(frame['left_rgb'],frame['right_rgb'],calibration,config)
    results=[]
    for i,identity in enumerate(frame['identity']):
        per={}
        for name,mask in regions.items():
            if mask is None:
                per[name]=dict(available=False,reason='source_not_recorded_or_unqualified');continue
            m=region_metrics(pred[i:i+1],target[i:i+1],valid[i:i+1],eligible[i:i+1],mask[i:i+1],calibration.K_left,
                disparity=frame['disparity_px'][i:i+1])
            roi=mask[i:i+1]&eligible[i:i+1]
            reasons={k:None if v is None else v[i:i+1] for k,v in replay['reasons'].items()}
            m['rejections']=rejection_summary(reasons,roi);m['available']=True;per[name]=m
            saved=valid[i:i+1];replayed=replay['valid'][i:i+1]
            m['saved_replay_validity']=dict(both_valid=int((roi&saved&replayed).sum()),
                saved_valid_replay_invalid=int((roi&saved&~replayed).sum()),
                saved_invalid_replay_valid=int((roi&~saved&replayed).sum()),both_invalid=int((roi&~saved&~replayed).sum()))
        samples=[]
        if wire is not None:
            roi=wire[i]&eligible[i]
            err=torch.where(roi&valid[i],(pred[i]-target[i]).abs(),-1.)
            tails=err.flatten().topk(min(max_cost_pixels//2,int((err>=0).sum()))).indices
            coords=roi.nonzero();count=min(max_cost_pixels-len(tails),len(coords))
            chosen=coords[torch.linspace(0,len(coords)-1,count).long()] if count else coords[:0]
            pixels=torch.cat((torch.stack((tails//pred.shape[-1],tails%pred.shape[-1]),1),chosen),0)
            pixels=torch.cat((torch.full((len(pixels),1),i),pixels),1).long()
            samples=sampled_cost_curves(frame['left_rgb'],frame['right_rgb'],pixels,
                list(range(config['min_disparity'],min(config['max_disparity'],calibration.width-1)+1)),config['block_size'],wire_mask=wire)
            for sample in samples:
                y,x=sample['row'],sample['column'];z=float(target[i,y,x])
                sample.update(identity=identity,frame_id=frame['frame_id'],reference_depth_m=z,
                    reference_disparity_px=float(calibration.K_left[0,0])*calibration.baseline_m/z,
                    saved_predicted_valid=bool(valid[i,y,x]),
                    saved_depth_m=float(pred[i,y,x]) if valid[i,y,x] else None,
                    left_gray_std=float(std[i,y,x]))
        results.append(dict(identity=identity,frame_id=frame['frame_id'],simulation_time=frame['simulation_time'],
            regions=per,cost_probes=samples,
            wire_gray_std=distribution(std[i][wire[i]&eligible[i]]) if wire is not None else None,
            replay_valid_mismatch_count=int((replay['valid'][i]!=valid[i]).sum())))
    sensitivity={}
    for tolerance in (.0005,.001,.002):
        masks=reference_visibility(target,reference.get('right_depth_m'),eligible,calibration.K_left,
            calibration.baseline_m,tolerance_m=tolerance)
        sensitivity[str(tolerance)]={name:None if mask is None else [int(row.sum()) for row in mask]
            for name,mask in masks.items()}
    return dict(per_env=results,reference_visibility_scope='same-frame_right_optical_Z_reprojection_proxy_not_right_instance_GT',
        visibility_tolerance_m=.001,right_wire_instance_mask=None,
        visibility_tolerance_sensitivity=sensitivity,replay_scope=replay['scope'],
        uniqueness_scope=replay.get('uniqueness_scope'),geometry_metadata=geometry_metadata),regions


def rejection_overlay(reasons,env_index):
    names=('low_texture_left','left_right_inconsistent','uniqueness_rejection')
    shape=next(value[env_index] for value in reasons.values() if isinstance(value,torch.Tensor))
    code=torch.zeros_like(shape,dtype=torch.float32);missing=[]
    for bit,name in enumerate(names):
        value=reasons.get(name)
        if value is None:missing.append(name)
        else:code+=value[env_index].float()*(2**bit)
    return code,missing


def write_visuals(output,frame,reference,calibration,regions,*,frame_report=None,replay=None):
    """Render actual saved pixels and ray-aligned point clouds; no GT completion."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from .tabletop_stereo_geometry import backproject
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cloud=backproject(frame['depth_m'],calibration.K_left,frame['valid']).numpy()
    truth=backproject(reference['depth_m'],calibration.K_left,reference['valid']).numpy()
    files=[]
    for i,identity in enumerate(frame['identity']):
        fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
        left=frame['left_rgb'][i].permute(1,2,0).numpy();right=frame['right_rgb'][i].permute(1,2,0).numpy()
        wire=regions.get('wire');mask=None if wire is None else wire[i].numpy()
        axes[0,0].imshow(left)
        if mask is not None and mask.any():axes[0,0].contour(mask,levels=[.5],colors='lime',linewidths=.5)
        axes[0,0].set_title('actual left RGB + recorded Wire ID mask' if mask is not None else 'actual left RGB; Wire ID GT unavailable')
        axes[0,1].imshow(right);axes[0,1].set_title('actual right RGB; right Wire ID GT NOT recorded')
        fields=[(frame['disparity_px'][i],'saved disparity (px)'),
            (float(calibration.K_left[0,0])*calibration.baseline_m/reference['depth_m'][i],'reference disparity fB/Z (px)'),
            ((frame['depth_m'][i]-reference['depth_m'][i]).abs()*1000,'absolute optical-Z error (mm)'),
            (frame['valid'][i].float(),'saved predicted validity'),
            (reference['depth_m'][i],'actual left reference optical-Z (m)')]
        for ax,(value,title) in zip([axes[0,2],axes[0,3],axes[1,0],axes[1,1],axes[1,2]],fields):
            kwargs=dict(cmap='turbo')
            if 'error' in title:kwargs.update(vmin=0,vmax=50)
            im=ax.imshow(value.numpy(),**kwargs);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.65)
        axes[1,3].imshow(frame['occlusion'][i].numpy(),cmap='gray')
        axes[1,3].set_title('baseline composite reject; NOT true occlusion GT')
        for ax in axes.flat:ax.set_axis_off()
        fig.suptitle(f"env_id={identity['env_id']} episode={identity['episode_uid']} frame={frame['frame_id']} t={frame['simulation_time']:.9f}; saved-native reanalysis only",fontsize=9)
        p=output/f"env{identity['env_id']}_frame{frame['frame_id']}_pixels.png";fig.savefig(p,dpi=110);plt.close(fig);files.append(p)
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for ax,image,label in zip(axes[0,:2],(frame['left_rgb'][i],frame['right_rgb'][i]),('left','right')):
            gray=image.mean(0);mean=F.avg_pool2d(gray[None,None],5,1,2)[0,0]
            std=(F.avg_pool2d(gray.square()[None,None],5,1,2)[0,0]-mean.square()).clamp_min(0).sqrt()
            im=ax.imshow(std.numpy(),cmap='magma',vmin=0,vmax=.01);fig.colorbar(im,ax=ax,shrink=.7)
            ax.set_title(label+' gray std (5x5 diagnostic; not estimator window)')
        ax=axes[0,2]
        if replay is not None:
            code,missing=rejection_overlay(replay['reasons'],i)
            ax.imshow(code.numpy(),vmin=0,vmax=7,cmap='tab10')
            ax.set_title('overlap: texture1 + LR2 + uniqueness4'+('\nNA: '+','.join(missing) if missing else ''))
        sample=None
        if frame_report is not None:
            probes=frame_report['per_env'][i]['cost_probes'];sample=probes[0] if probes else None
        if sample is not None:
            gt=sample['reference_disparity_px'];d=sample['disparities_px']
            for ax,key in zip(axes[1],('sad_cost','census_hamming','horizontal_gradient_cost')):
                ax.plot(d,[np.nan if x is None else x for x in sample[key]],linewidth=1)
                ax.axvline(gt,color='green',label='reference fB/Z');ax.set_title(key+f" @ y{sample['row']} x{sample['column']}")
                ax.set_xlim(max(min(d),gt-12),min(max(d),gt+12));ax.grid();ax.set_xlabel('disparity (px)')
            axes[1,0].legend(fontsize=6)
        else:
            for ax in axes[1]:ax.text(.1,.5,'Wire source-qualified cost probe unavailable');ax.set_axis_off()
        fig.suptitle(f"env {identity['env_id']} frame {frame['frame_id']}: diagnostic cost, not a candidate algorithm result")
        p=output/f"env{identity['env_id']}_frame{frame['frame_id']}_cost_texture.png";fig.savefig(p,dpi=110);plt.close(fig);files.append(p)
        if mask is None:continue
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        gt=truth[i][mask];pr=cloud[i][mask&frame['valid'][i].numpy()]
        for ax,(a,b) in zip(axes,((0,1),(0,2),(1,2))):
            ax.scatter(gt[:,a],gt[:,b],s=1,label='reference-visible surface',alpha=.5)
            ax.scatter(pr[:,a],pr[:,b],s=1,label='saved stereo same-pixel rays',alpha=.5)
            ax.set_xlabel('XYZ'[a]+' optical (m)');ax.set_ylabel('XYZ'[b]+' optical (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid()
        axes[0].legend(fontsize=6)
        fig.suptitle('Visible Wire surface; no nearest-neighbour realignment or hidden-node completion')
        p=output/f"env{identity['env_id']}_frame{frame['frame_id']}_wire3d.png";fig.savefig(p,dpi=120);plt.close(fig);files.append(p)
    return [dict(path=str(p),sha256=_sha(p)) for p in files]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-cost-pixels',type=int,default=24)
    parser.add_argument('--no-visuals',action='store_true')
    args=parser.parse_args(argv)
    if args.threads<1 or not 0<=args.max_cost_pixels<=128:raise ValueError('bounded threads/cost pixels required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    results=[];failures=[];cache={};hashes={};start=time.perf_counter()
    for index,row in enumerate(manifest['results']):
        key=f"{index:02d}_{row['experiment']}_{row['width']}x{row['height']}"
        folder=args.output/key;folder.mkdir()
        try:
            report,proof_path,parent_identity=resolve_proof(row)
            hashes[str(report)]=_sha(report);hashes[str(proof_path)]=_sha(proof_path)
            capture,refs,c,config,p=load_capture(proof_path)
            if (c.width,c.height)!=(row['width'],row['height']) or capture['cohort_sha256']!=row['cohort_sha256']:
                raise ValueError('raw capture does not match requested benchmark row')
            original_bindings=[p[k] for k in ('capture_artifact','calibration_artifact','validation_config_artifact',
                'performance_artifact','completion_artifact')]+p['reference_artifacts']+list(p['producer'][k] for k in ('source_artifact','config_artifact'))
            original_bindings += [ref['wire_mask_source_artifact'] for ref in refs if ref.get('wire_mask_source_artifact')]
            for binding in original_bindings:
                checked_artifact(binding);hashes[binding['path']]=binding['sha256']
            frame_reports=[];figures=[];unique=[]
            for fi,(frame,ref) in enumerate(zip(capture['frames'],refs)):
                pair_hash=hashlib.sha256(frame['left_rgb'].numpy().tobytes()+frame['right_rgb'].numpy().tobytes()).hexdigest()
                cache_key=pair_hash+json.dumps(config,sort_keys=True)+c.sha256
                new_pair=cache_key not in cache
                if new_pair:
                    cache[cache_key]=replay_depth_conditions(frame['left_rgb'],frame['right_rgb'],c,config)
                    unique.append(fi)
                info,regions=analyze_frame(frame,ref,c,config,replay=cache[cache_key],
                    max_cost_pixels=args.max_cost_pixels if new_pair else 0)
                info.update(frame_index=fi,rgb_pair_sha256=pair_hash,
                    cost_probe_reuse_note=None if new_pair else 'bit-identical RGB/config/calibration replay reused;this frame metrics/identity still recomputed')
                frame_reports.append(info)
                if fi==0 and not args.no_visuals:
                    figures=write_visuals(folder/'figures',frame,ref,c,regions,frame_report=info,replay=cache[cache_key])
            item=dict(experiment=row['experiment'],resolution_wh=[c.width,c.height],
                scope='offline_saved_native_artifact_reanalysis;no_new_native_capture',
                proof_path=str(proof_path),proof_sha256=_sha(proof_path),calibration_sha256=c.sha256,
                resolved_matcher_config=config,cohort_sha256=capture.get('cohort_sha256'),
                frames=frame_reports,figures=figures,unique_replay_frame_indices=unique,
                input_bindings={k:p[k] for k in ('capture_artifact','calibration_artifact','reference_artifacts')},
                producer=p['producer'],accuracy_qualification_granted=False,physical_qualification_granted=False)
            item.update(diagnostic_implementation=implementation_identity(),parent_identity=parent_identity)
            _write(folder/'diagnostics.json',item)
            results.append(dict(key=key,path=str(folder/'diagnostics.json'),sha256=_sha(folder/'diagnostics.json'),
                frame_count=len(frame_reports),env_observations=sum(len(f['per_env']) for f in frame_reports),
                unique_replay_frame_indices=unique))
            print(json.dumps(dict(completed=key,frames=len(frame_reports),unique_replays=len(unique))),flush=True)
        except (ValueError,FileNotFoundError,KeyError,RuntimeError) as error:
            failure=dict(key=key,error_type=type(error).__name__,message=str(error),expected_report_artifact=row['report_artifact'])
            failures.append(failure);_write(folder/'failure.json',failure)
            print(json.dumps(failure),flush=True)
    unchanged={p:_sha(p)==sha for p,sha in hashes.items()}
    if not all(unchanged.values()):raise RuntimeError('original input modified during diagnostics')
    import platform
    summary=dict(version='dh116-stereo-j1-error-diagnostics-v1',scope='offline_reanalysis_of_existing_native_raw;not_new_native_validation',
        diagnostic_implementation=implementation_identity(),
        input_manifest=dict(path=str(args.manifest),sha256=_sha(args.manifest)),python=platform.python_version(),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,device='CPU',threads=args.threads,
        completed_runs=results,failed_runs=failures,wall_seconds=time.perf_counter()-start,
        new_native_capture_count=0,training_started=False,accuracy_qualified=False,
        physical_task_qualified=False,deployment_qualified=False,original_report_and_proof_hashes_unchanged=unchanged)
    _write(args.output/'summary.json',summary)
    print(json.dumps(dict(completed_runs=len(results),failed_runs=len(failures),wall_seconds=summary['wall_seconds'])),flush=True)
    return 1 if failures else 0
