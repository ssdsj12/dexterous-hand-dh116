"""Exclusive full-state checkpoint derivation for one validated reward stage."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_surface_contract import validate_contract,semantic_sha256
from go2_pvcnn.control.dh116.tabletop_surface_imitation import validate_dataset
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ablation_contract


def identical(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--reward-profile',default='baseline')
    p.add_argument('--corrections',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    oldsha=sha(a.input);source=torch.load(a.input,map_location='cpu',weights_only=False)
    validate_contract(source['contract'])
    derived=copy.deepcopy(source)
    if a.reward_profile!='baseline':
        derived['contract']['surface_reward_ablation']=ablation_contract(a.reward_profile)
    else:derived['contract'].pop('surface_reward_ablation',None)
    derived['contract']['surface_contract_sha256']=semantic_sha256(derived['contract'])
    validate_dataset(torch.load(a.demonstrations,map_location='cpu',weights_only=False),derived['contract'])
    if 'surface_imitation' not in derived:raise ValueError('full BC/PPO state required')
    derived['surface_imitation']['dataset_sha256']=sha(a.demonstrations)
    if a.corrections:
        if a.reward_profile!='baseline':raise ValueError('one factor per stage: corrections or reward')
        from go2_pvcnn.control.dh116.tabletop_surface_dagger import CorrectiveRegularizer
        CorrectiveRegularizer(torch.load(a.corrections,map_location='cpu',weights_only=False),derived['contract'],'cpu')
        derived['surface_corrections']=dict(dataset_sha256=sha(a.corrections),batch_size=32,
            update='after_PPO_and_original_BC;shared_retained_BC_Adam',ppo_storage_used=False)
    unchanged={k:identical(v,derived[k]) for k,v in source.items()
        if k not in ('contract','surface_imitation','surface_corrections')}
    if not all(unchanged.values()):raise RuntimeError('non-metadata checkpoint state changed')
    manifest=dict(source=str(a.input),source_sha256=oldsha,demonstrations_sha256=sha(a.demonstrations),
        reward_profile=a.reward_profile,contract_sha256=derived['contract']['surface_contract_sha256'],
        invariant_checks=unchanged,
        retained='all_policy_PPO_Adam_BC_Adam_RNG_counters;only_contract_and_demo_source_metadata_change')
    if a.output.resolve()==a.input.resolve():raise ValueError('source must never be overwritten')
    sidecar=a.output.with_suffix('.json')
    if a.output.exists() or sidecar.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as out:torch.save(derived,out)
    manifest['derived_sha256']=sha(a.output)
    if sha(a.input)!=oldsha:raise RuntimeError('source changed during derivation')
    with sidecar.open('x') as out:json.dump(manifest,out,indent=2)
    print(json.dumps(manifest))


if __name__=='__main__':main()
