"""Assemble contract-checked Teacher labels for Student's visited failures."""
import argparse
import json
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_surface_dagger import (
    build_failure_relabels, load_evaluation)


parser=argparse.ArgumentParser()
parser.add_argument('--student-evaluation',type=Path,required=True)
parser.add_argument('--teacher-actions',type=Path,required=True,
                    help='24D labels produced at the recorded Student failure states')
parser.add_argument('--contract',type=Path,required=True)
parser.add_argument('--source-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
evaluation=load_evaluation(args.student_evaluation)
labels=torch.load(args.teacher_actions,map_location='cpu',weights_only=False)
if isinstance(labels,dict):
    labels=labels.get('teacher_actions',labels.get('actions'))
contract=json.loads(args.contract.read_text(encoding='utf-8'))
payload=build_failure_relabels(evaluation,labels,contract,
    source_sha256=args.source_sha256,output=args.output)
print(json.dumps({k:payload[k] for k in ('version','sample_count','contract_sha256',
    'source_sha256','execution_owner','teacher_servo_used')},sort_keys=True))
