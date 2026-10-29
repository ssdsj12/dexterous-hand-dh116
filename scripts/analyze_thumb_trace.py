import json
import sys
from pathlib import Path
import torch

p=Path(sys.argv[1])
x=torch.load(p/'thumb_physics_trace.pt',map_location='cpu',weights_only=True)['samples'][-480:]
report=json.loads((p/'thumb_probe.json').read_text())
for i,row in enumerate(report['cases']):
    s=x[:,i]
    masters=s[:,14:26][:,[1,2,7,8]]
    followers=s[:,4:14][:,[0,1,5,6]]+masters*masters.new_tensor([.88178789,.9206,.88178789,.9206])
    follower_speed=torch.diff(followers,dim=0)*480
    f=s[:,:4]-s[:,:4].mean(0)
    power=torch.fft.rfft(f,dim=0).abs().square();power[0]=0.
    frequencies=torch.fft.rfftfreq(len(f),1/480)
    print(json.dumps(dict(case=i,failures=row['failures'],
        thumb_force_range_n=(s[:,:4].amax(0)-s[:,:4].amin(0))[[0,2]].tolist(),
        master_angle_range_rad=(s[:,14:26].amax(0)-s[:,14:26].amin(0))[[1,2,7,8]].tolist(),
        target_range_rad=(s[:,26:38].amax(0)-s[:,26:38].amin(0))[[1,2,7,8]].tolist(),
        mimic_range_rad=(s[:,4:14].amax(0)-s[:,4:14].amin(0))[[0,1,5,6]].tolist(),
        max_mimic_rad=s[:,4:14].abs().amax().item(),
        follower_speed_peak=follower_speed.abs().amax(0).tolist(),
        force_peak_hz=frequencies[power.argmax(0)].tolist(),
        force_power_above_180hz=(power[frequencies>180].sum(0)/power.sum(0).clamp_min(1e-9)).tolist(),
        follower_speed_over_2_8_fraction=(follower_speed.abs()>2.8).double().mean(0).tolist()),allow_nan=False))
