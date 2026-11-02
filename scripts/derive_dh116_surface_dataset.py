"""Correct the 371D digest after end-support metadata finalization, exclusively.

Observations/actions/qualification stay byte-equivalent tensors. This is an
explicit metadata derivation of an existing native replay, not a new replay.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256,validate_contract
from go2_pvcnn.control.dh116.tabletop_surface_imitation import validate_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sidecar=a.output.with_suffix('.json')
    if a.output.resolve()==a.input.resolve() or a.output.exists() or sidecar.exists():
        raise FileExistsError('a distinct exclusive derived artifact is required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    before=sha(a.input);old=torch.load(a.input,map_location='cpu',weights_only=False)
    new=copy.deepcopy(old);previous=new['contract']['surface_contract_sha256']
    new['contract']['surface_contract_sha256']=semantic_sha256(new['contract'])
    validate_contract(new['contract']);validate_dataset(new)
    for key in ('observations','actions','episode_ids'):
        if not torch.equal(old[key],new[key]):raise RuntimeError('native sample changed')
    with a.output.open('xb') as stream:torch.save(new,stream)
    report=dict(source_sha256=before,derived_sha256=sha(a.output),
        previous_contract_sha256=previous,contract_sha256=new['contract']['surface_contract_sha256'],
        changed='digest_only_after_end_support_metadata;native_samples_and_proof_unchanged',
        samples=int(len(new['actions'])),new_native_replay=False)
    if sha(a.input)!=before:raise RuntimeError('source changed')
    with sidecar.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
