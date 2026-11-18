"""Small measured-force teacher sweep using calibrated closing directions."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
initial=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_v1.json').read_text())
direction=sum((initial['diagnostics']['hands'][side]['closure_direction_active_unit'] for side in ('left','right')),[])
cases=[]
for depth in (-.0005,-.001,-.0015,-.002):
    for extra in (.005,.015):
        cases.append(dict(name=f'below_z{abs(depth)*1000:g}mm_close{extra*1000:g}mrad_force2p5N',
            contact_z_m=depth,force_n=[2.5,2.5],joint_offset_rad=[v*extra for v in direction],
            xy_m=[[0.,0.],[0.,0.]],closing_ramp_s=1.))
path=root/'configs/dh116_training/teacher_below_axis_8cases_v1.json'
if path.exists():raise ValueError('preserve existing teacher cases')
path.write_text(json.dumps(dict(format='below_axis_force_teacher_cases_v1',cases=cases),indent=2)+'\n')
print(path)
