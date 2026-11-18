"""Save a measured close row for stage-local initialization, not grasp credit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod
from go2_pvcnn.control.dh116.close_initialization import from_contact_row,rebase_hand_stiffness
from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
from go2_pvcnn.control.dh116.mpc_feedback import estimate_tracking_bias
from go2_pvcnn.control.dh116.wire_table import TABLE_CENTER, TABLE_SIZE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--contact-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-step',type=int,help='Use a specific already validated close row')
    parser.add_argument('--hand-stiffness',type=float,
        help='Optional new simulation Kp; rebase targets to preserve initial proportional torque')
    args=parser.parse_args()
    manifest=json.loads((args.source_run/'run_manifest.json').read_text())
    config=manifest['run_config']
    if config['rod_backend']!='torch_cuda' or config['grasp_entry']!='close':
        raise ValueError('this initializer requires the production direct-close GPU run')
    row=None;controller_reference=None
    with (args.source_run/'trajectory.jsonl').open() as stream:
        for line in stream:
            try:current=json.loads(line)
            except json.JSONDecodeError:continue  # live writer can have one trailing partial row
            if args.source_step is not None and current['step']>args.source_step:break
            row=current
            sample=current.get('hand_mpc_training_sample')
            if sample is not None and current.get('passed'):
                controller_reference=dict(source_step=current['step'],
                    commanded_penetration_m=sample['commanded_penetration_m'])
                bias=sample.get('planner_diagnostics',{}).get('tracking_bias_rad')
                if bias is not None:controller_reference['tracking_bias_rad']=bias
    if row is None:raise ValueError('no physical row found')
    if args.source_step is not None and row['step']!=args.source_step:
        raise ValueError('requested physical source step is unavailable')
    decisions=args.source_run/'contact_decisions.jsonl'
    if controller_reference is not None and decisions.exists():
        with decisions.open() as stream:
            records=[json.loads(line) for line in stream if line.endswith('\n')]
        controller_reference['tracking_bias_rad']=estimate_tracking_bias(records,through_step=row['step']).tolist()
    sha=hashlib.sha256(args.contact_manifest.read_bytes()).hexdigest()
    if sha!=config['contact_manifest_sha256']:raise ValueError('source contact manifest changed')
    rod=TorchCudaCosseratRod(GpuRodConfig(n_elem=40,base_length=.5,base_radius=.008,
        density=1000.,youngs_modulus=1e6,shear_modulus_ratio=1.5,dt=config['wire_py_dt'],damping=.01,
        rod_start=(.43,-.25,.308),rod_direction=(0.,1.,0.),rod_normal=(0.,0.,1.),
        table_center=tuple(TABLE_CENTER),table_size=tuple(TABLE_SIZE),
        contact_stiffness=1e4,contact_damping=10.,friction_coefficient=.5,
        tangential_damping=config['contact_tangential_damping'],
        contact_subsamples=config['contact_subsamples']),device=config['device'])
    for channel,record in load_mesh_manifest(args.contact_manifest,sha).items():
        mesh=o3d.io.read_triangle_mesh(str(record['resolved_path']))
        rod.register_contact_mesh(channel,np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    payload=from_contact_row(row,source_run=args.source_run,
        asset_sha256=manifest['asset_manifest']['asset_sha256'],contact_manifest_sha256=sha,
        rod_physical=rod.export_physical_state(),controller_reference=controller_reference)
    if args.hand_stiffness is not None:
        payload=rebase_hand_stiffness(payload,old_stiffness=config['hand_stiffness'],
                                     new_stiffness=args.hand_stiffness)
    rod.restore_physical_state(payload['rod'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(payload,stream,allow_nan=False)
    print(json.dumps(dict(path=str(args.out),sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        source=payload['source'],grasp_qualified=False)))


if __name__=='__main__':
    main()
