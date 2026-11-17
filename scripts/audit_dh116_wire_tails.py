"""Read-only same-cohort measured-pixel provenance and >5 mm tail preflight."""
import argparse
import json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture, _label_masks
from go2_pvcnn.control.dh116.tabletop_stereo_wire_matching import wire_colour_mask
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import (
    validate_surface_samples, tail_geometry, optical_sample, window_count,
    rejection_at_pixel, completeness_counts)


def read_bound(path, sha):
    if file_sha256(path) != sha:
        raise ValueError(f'input SHA mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def write_jsonl(path, rows):
    with path.open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof', type=Path, required=True)
    p.add_argument('--depth-replay', type=Path, required=True)
    p.add_argument('--sample-replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(argv)
    torch.set_num_threads(2)
    capture, references, calibration, config, proof=load_capture(args.proof)
    ds_path=args.depth_replay/'summary.json'
    ss_path=args.sample_replay/'summary.json'
    ds=json.loads(ds_path.read_text()); ss=json.loads(ss_path.read_text())
    if (ds['capture_sha256']!=proof['capture_artifact']['sha256']
            or ds['calibration_sha256']!=calibration.sha256
            or ds['reference_sha256']!=[r['sha256'] for r in proof['reference_artifacts']]
            or ss['capture_sha256']!=ds['capture_sha256']
            or ss['proof_sha256']!=file_sha256(args.proof)
            or ss['depth_replay_summary_sha256']!=file_sha256(ds_path)):
        raise ValueError('capture/reference/calibration/sampling chain mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    samples=[]; tails=[]; frames=[]
    for i,(frame,ref) in enumerate(zip(capture['frames'],references)):
        db=ds['frames'][i]['methods']['wire_silhouette']
        sb=ss['frames'][i]['methods']['96']
        depth=read_bound(args.depth_replay/db['pixel_maps_file'],db['pixel_maps_sha256'])
        saved=read_bound(args.sample_replay/sb['pixel_file'],sb['sha256'])
        if ss['frames'][i]['depth_pixel_sha256']!=db['pixel_maps_sha256']:
            raise ValueError('sample-to-depth map mismatch')
        points=dict(xyz=saved['xyz'], point_pixels=saved['point_pixels'],
            point_valid=saved['valid'],point_provenance=saved['provenance'])
        points.update({key:saved[key] for key in ('component_id','track_id','image_track_id') if key in saved})
        z=depth['stereo_depth_m']; K=frame.get('K_left',calibration.K_left)
        sampling=validate_surface_samples(frame['left_rgb'],z,points,K,
            frame.get('T_world_left',calibration.T_world_left),radius=3)
        left_mask=wire_colour_mask(frame['left_rgb']); right_mask=wire_colour_mask(frame['right_rgb'])
        finger=_label_masks(ref,frame)['finger']
        envrows=[]
        for env,identity in enumerate(frame['identity']):
            denominator=int((ref['wire_reference_mask'][env]&ref['valid'][env]).sum())
            audit=sampling[env]; count_before=len(tails); correct=0
            for slot in points['point_valid'][env].nonzero().flatten().tolist():
                y,x=points['point_pixels'][env,slot].tolist()
                km=K[env] if K.ndim==3 else K
                zr=float(ref['depth_m'][env,y,x])
                if not torch.isfinite(torch.tensor(zr)) or zr<=0:
                    raise ValueError('sample has no finite optical ground truth; do not invent tail error')
                geometry=tail_geometry(float(z[env,y,x]),zr,km,y=y,x=x,
                    disparity=float(depth['disparity_px'][env,y,x]),baseline=calibration.baseline_m)
                source=next(s for s in audit['samples'] if s['y']==y and s['x']==x)
                native_wire=bool(ref['wire_reference_mask'][env,y,x])
                correct+=int(native_wire and not geometry['is_tail_5mm'])
                row=dict(identity=identity,frame_index=i,frame_id=frame['frame_id'],
                    sample_slot=slot,left_pixel_yx=[y,x],source=source,geometry=geometry,
                    left_native_wire=native_wire,source_pixel_sha256=db['pixel_maps_sha256'],
                    sampling_pixel_sha256=sb['sha256'],reference_sha256=proof['reference_artifacts'][i]['sha256'])
                samples.append(row)
                if not geometry['is_tail_5mm']:
                    continue
                rx=geometry['right_match_x_px']; gx=geometry['right_reference_x_px']
                rz=optical_sample(ref['right_depth_m'][env],y,rx)
                gz=optical_sample(ref['right_depth_m'][env],y,gx)
                reasons=rejection_at_pixel(depth['rejection_flags'],env,y,x)
                row=dict(row,windows=dict(radius_px=config['block_size']//2,
                    left_rgb_wire=window_count(left_mask[env],y,x,config['block_size']//2),
                    left_native_wire=window_count(ref['wire_reference_mask'][env],y,x,config['block_size']//2),
                    left_native_finger=window_count(finger[env],y,x,config['block_size']//2),
                    right_rgb_wire_at_predicted=window_count(right_mask[env],y,rx,config['block_size']//2),
                    right_rgb_wire_at_reference=window_count(right_mask[env],y,gx,config['block_size']//2),
                    right_native_wire=None,right_native_finger=None),
                    accepted_intersections={key:bool(depth[key][env,y,x]) for key in (
                        'wire_mask','wire_left_depth_valid','wire_left_right_consistent',
                        'wire_depth_range_valid','native_wire_reference_mask','native_wire_right_occlusion_proxy',
                        'wire_candidate_centerline','wire_candidate_centerline_band')},
                    overlapping_rejection_flags=reasons,
                    right_optical_reference=dict(at_predicted_match_m=rz,at_left_gt_projection_m=gz,
                        projected_gt_minus_right_depth_m=None if gz is None else zr-gz,
                        predicted_z_minus_right_depth_m=None if rz is None else float(z[env,y,x])-rz),
                    cause_status='ambiguous_not_unique_causality',
                    limitations=['right_native_instance_ID_absent',
                        'right_optical_depth_does_not_identify_wire_or_occluder',
                        'left_window_finger_overlap_is_not_causal_proof',
                        'window_values_are_recorded_matching_window_support_not_hidden_geometry'])
                tails.append(row)
            envrows.append(dict(env_id=identity['env_id'],grasp_style=identity['grasp_style'],
                frame_id=frame['frame_id'],**completeness_counts(ref['wire_reference_mask'][env],
                    ref['valid'][env],correct,audit['unique_pixels']),tail_count=len(tails)-count_before,
                sampling_audit={key:value for key,value in audit.items() if key!='samples'}))
        frames.append(dict(frame_id=frame['frame_id'],environments=envrows))
    write_jsonl(args.output/'all_sample_lineage.jsonl',samples)
    write_jsonl(args.output/'tails_over_5mm.jsonl',tails)
    result=dict(scope='existing_native_RGB_offline_tail_preflight',native_capture_started=False,
        capture_sha256=ds['capture_sha256'],proof_sha256=file_sha256(args.proof),
        calibration_sha256=calibration.sha256,depth_summary_sha256=file_sha256(ds_path),
        sample_summary_sha256=file_sha256(ss_path),sample_count=len(samples),tail_count=len(tails),
        unique_tail_env_pixel_count=len({(r['identity']['env_id'],*r['left_pixel_yx']) for r in tails}),
        physical_initial_conditions=len({r['identity']['physical_initial_condition_sha256'] for r in samples}),
        threshold_m=.005,metric='Euclidean_3D_error_at_same_left_pixel',
        default_configuration_changed=False,interpolation_created_points=False,
        all_sample_lineage_sha256=file_sha256(args.output/'all_sample_lineage.jsonl'),
        tails_sha256=file_sha256(args.output/'tails_over_5mm.jsonl'),frames=frames,
        source_sha256={str(Path(__file__).name):file_sha256(Path(__file__)),
            'tabletop_stereo_wire_tail_audit.py':file_sha256(Path(__file__).resolve().parents[1]/
                'src/go2_pvcnn/control/dh116/tabletop_stereo_wire_tail_audit.py')})
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(samples=len(samples),tails=len(tails),output=str(args.output))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
