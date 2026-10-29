"""Replay recorded measurements through guards, not a simulator-state replay."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_env import physical_guard

root=Path(sys.argv[1])
trace=json.loads((root/'physics_fault_trace.json').read_text())
monitor=json.loads((root/'monitor.json').read_text())
detail=ast.literal_eval(monitor['error'].rsplit('tensor physical safety failure: ',1)[1].split('\n',1)[0])
limits=torch.tensor(detail['joint_limits_rad'],dtype=torch.float64)
checked=0
for frame in trace['frames']:
    tensor=lambda key:torch.tensor(frame[key],dtype=torch.float64)
    world=SimpleNamespace(n=1,device='cpu',q=tensor('joint_position_rad'),
        qd=tensor('joint_velocity_rad_s'),limits=limits,errors=tensor('mimic_errors'))
    m={key:tensor(key) for key in ('object_link_loads','hand_object_loads','mimic_errors','clearance','bottom')}
    m['joint_failure']=torch.tensor([detail['task_failure'][0]])
    strict,rf,rm=physical_guard(world,m)
    force=float(m['object_link_loads'].max());mimic=float(m['mimic_errors'].abs().max())
    old_reset=(30.<force<45. and mimic<=.02) or (.02<mimic<.03 and force<=30.)
    if frame is trace['frames'][-1]:
        assert not strict.any() and rf.all() and rm.all() and not old_reset
        print(json.dumps(dict(replay='recorded_measurements_only',frames=len(trace['frames']),
            force_n=force,mimic_rad=mimic,strict_safe=False,
            old_reset_eligible=old_reset,new_isolated_failed_reset_eligible=True)))
    checked+=1
assert checked==len(trace['frames'])
