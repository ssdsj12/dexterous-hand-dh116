"""Print standalone command; --run executes only when no DH116 verifier exists."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',action='store_true');p.add_argument('--output',default='runs/table_margin_trial1')
p.add_argument('--steps',type=int,default=160000);p.add_argument('--gui',action='store_true')
a=p.parse_args();cfg=root/'configs/table_margin_raise1mm'
pins=json.loads((cfg/'validation.json').read_text())['files']
for name,digest in pins.items():
    if hashlib.sha256((cfg/name).read_bytes()).hexdigest()!=digest:raise ValueError('configuration changed: '+name)
output=(root/a.output).resolve()
if not output.is_relative_to(root/'runs'):raise ValueError('output must be under project runs')
if output.exists():raise FileExistsError(output)
cmd=[sys.executable,str(root/'scripts/verify_fixed_dual_panda_dh116.py'),
    '--asset-dir',str(root/'assets/fixed_dual_panda_dh116_v1'),
    '--active-pose',str(cfg/'prepose.json'),'--active-pose-sha256',pins['prepose.json'],
    '--approach-certificate',str(cfg/'approach.json'),'--approach-certificate-sha256',pins['approach.json'],
    '--point-references',str(cfg/'point_references.json'),
    '--contact-manifest',str(cfg/'contact/manifest.json'),'--contact-manifest-sha256',pins['contact/manifest.json'],
    '--backend-root',str(root/'vendor/DeformX'),'--grasp-sequence','--coordinated-closure',
    '--closure-loads','1.0005','.8855','1.012','1.0695','--contact-subsamples','16',
    '--contact-tangential-damping','1000','--clearance-certificate-cache','--contact-broad-phase',
    '--gravity-hold','--arm-effort-limit','60','--hand-effort-limit','.2','--hand-stiffness','10',
    '--hand-damping','.2','--external-force-iterations','--motion-prediction-order','1',
    '--physics-dt','.0001','--steps',str(a.steps),'--solver-position-iterations','1',
    '--table-wire','--save-stage-states','--device','cuda:0','--output',str(output)]
if not a.gui:cmd.append('--headless')
print(shlex.join(cmd),flush=True)
if a.run:
    running=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    if any('verify_fixed_dual_panda_dh116.py' in line for line in running.splitlines()):
        raise RuntimeError('DH116 verifier already exists; keep a single Isaac instance')
    os.execv(sys.executable,cmd)
