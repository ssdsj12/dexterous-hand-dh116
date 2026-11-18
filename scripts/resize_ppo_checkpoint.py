"""Create a separate, audited checkpoint for a smaller PPO environment batch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch


def resize(source,count):
    old=source['contract']['num_envs']
    if not isinstance(count,int) or not 1<=count<old:
        raise ValueError('new environment count must be a positive strict reduction')
    for name in ('episodes','successes','recovered'):
        if name not in source or source[name].shape!=(old,):
            raise ValueError('invalid per-environment counter: '+name)
    result=copy.deepcopy(source)
    result['contract']['num_envs']=count
    result['contract']['minibatches']=min(8,count)
    for name in ('episodes','successes','recovered'):
        result[name]=result[name][:count].clone()
    report=dict(old_num_envs=old,new_num_envs=count,iteration=source['iteration'],
        timesteps=source['timesteps'],policy_retained=True,optimizer_retained=True,
        per_env_counters='first_n_slots_retained; physical_scenes_reset',
        original_counter_totals={name:int(source[name].sum()) for name in ('episodes','successes','recovered')},
        rollout_storage='recreated_by_trainer_for_new_batch',reward_and_physics_unchanged=True)
    result['environment_resize']=report
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--num-envs',type=int,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    result,report=resize(source,args.num_envs)
    report['source']=str(args.source)
    report['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
    with args.output.open('xb') as stream:torch.save(result,stream)
    args.output.with_suffix('.resize.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
