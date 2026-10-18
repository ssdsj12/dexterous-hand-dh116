"""Record a pinned comparison of no-reaction and reacting physics evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from go2_pvcnn.control.dh116.reaction_pair import compare_pair

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reacting',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refusing to overwrite evidence')
    summaries=[];trajectories=[];pins={}
    for directory in (a.baseline,a.reacting):
        for file in ('summary.json','trajectory.jsonl'):
            path=(directory/file).resolve();payload=path.read_bytes()
            pins[str(path)]=hashlib.sha256(payload).hexdigest()
            if file=='summary.json':summaries.append(json.loads(payload))
            else:trajectories.append([json.loads(line) for line in payload.splitlines()])
    report=compare_pair(*summaries,*trajectories);report['evidence_sha256']=pins
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
