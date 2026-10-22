"""Train teacher/student residual policies from measured full-scene MPC runs."""
import argparse
import hashlib
import json
from pathlib import Path

from go2_pvcnn.control.dh116.contact_imitation import fit_contact_pair
from go2_pvcnn.control.dh116.contact_replay import load_contact_run,load_complete_contact_run
from go2_pvcnn.control.dh116.complete_contact_policy import fit_complete_pair
from go2_pvcnn.control.dh116.evidence_journal import EvidenceJournal
from go2_pvcnn.control.dh116.teacher_student import write_replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--allow-incomplete',action='store_true',
        help='Explicit partial-action diagnostic; never labels a grasp/lift as successful')
    parser.add_argument('--legacy-contact-replay',action='store_true',
        help='Explicit old 42/78-input diagnostic format; default requires complete v2 intervals')
    args=parser.parse_args()
    data=[];sources=[];episodes=[]
    for path in args.source_run:
        loader=load_contact_run if args.legacy_contact_replay else load_complete_contact_run
        rows,metadata=loader(path,require_success=not args.allow_incomplete)
        if sources and abs(metadata['control_dt_s']-sources[0]['control_dt_s'])>1e-12:
            raise ValueError('source runs must share the same hand-control interval')
        data.extend(rows);episodes.append(rows);sources.append(metadata)
    if args.legacy_contact_replay:
        models,metrics=fit_contact_pair(data,control_dt=sources[0]['control_dt_s'],
            epochs=args.epochs,device=args.device,seed=args.seed)
    else:
        models,metrics=fit_complete_pair(episodes,control_dt=sources[0]['control_dt_s'],
            epochs=args.epochs,device=args.device,seed=args.seed)
    qualified=all(source['grasp_lift_passed'] for source in sources)
    args.out.mkdir(parents=True,exist_ok=False)
    replay={};checkpoints={}
    if not args.legacy_contact_replay:
        replay_path=args.out/'complete_replay.jsonl'
        with EvidenceJournal(replay_path) as journal:
            for source,episode in zip(sources,episodes):
                for row in episode:journal.append(dict(row,source_run=source['source_run']))
        replay['complete']=dict(path=str(replay_path),intervals=len(data),
            sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest())
    for role,policy in models.items():
        policy.training_metadata=dict(
            training_scope='successful_grasp_trajectory' if qualified else 'partial_action_diagnostic',
            validated_source_grasp_lift=qualified,source_runs=[source['source_run'] for source in sources])
        if args.legacy_contact_replay:
            replay[role]=write_replay(args.out/(role+'.jsonl'),data,role=role)
        checkpoint=args.out/(role+'.pt');policy.save(checkpoint)
        checkpoints[role]=dict(path=str(checkpoint),input_dim=policy.input_dim,action_dim=12,
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    manifest=dict(schema='dh116-physical-contact-imitation-training-v1',sources=sources,
        training_scope='successful_grasp_trajectory' if qualified else 'partial_action_diagnostic',
        validated_source_grasp_lift=qualified,student_isaac_validated=False,
        model=('MLP_64_64_residual_to_previous_action' if args.legacy_contact_replay else
               'MLP_128_128_feedback_history_residual_to_previous_action'),
        feature_schema='legacy_42_78' if args.legacy_contact_replay else 'dh116-contact-decision-v2',
        student_geometry_source='simulation_observations_not_real_vision',arm_state_included=False,
        rod_backend='torch_cuda',device=args.device,metrics=metrics,replay=replay,checkpoints=checkpoints)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),training_scope=manifest['training_scope'],
        validated_source_grasp_lift=qualified,student_isaac_validated=False,**metrics)))


if __name__=='__main__':main()
