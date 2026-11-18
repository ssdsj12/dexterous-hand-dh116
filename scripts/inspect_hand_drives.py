import json
from pathlib import Path
from pxr import Usd,UsdPhysics

manifest=json.loads(Path('assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
for source in manifest['sources']:
    if source['kind']!='dh116':continue
    stage=Usd.Stage.Open(source['path'])
    rows=[]
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            drive=UsdPhysics.DriveAPI(p,'angular')
            rows.append(dict(name=p.GetName(),drive_type=drive.GetTypeAttr().Get(),
                kp=drive.GetStiffnessAttr().Get(),kd=drive.GetDampingAttr().Get()))
    print(source['side'],json.dumps(rows))
