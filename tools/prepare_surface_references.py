"""Revalidate every archived surface point against the relocated USD."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd
from tools.migration_copy import sha256
from go2_pvcnn.control.dh116.usd_surface import read_surface,layer_pins
from go2_pvcnn.control.dh116.contact_frames import surface_reference

root=Path(__file__).resolve().parents[1]
path=root/'configs/surface_references.json'
if path.exists():raise FileExistsError(path)
asset=root/'assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd'
stage=Usd.Stage.Open(str(asset))
data=json.loads((root/'archive/dh116_v2/surface_references_v1/references.json').read_text())
for ref in data['references']:
    surface=read_surface(stage,ref['body_path'])
    triangle=np.asarray(surface['points_body'])[np.asarray(surface['triangles'])[ref['triangle_index']]]
    point,normal=surface_reference(triangle,ref['weights'])
    if not np.allclose(point,ref['point'],atol=1e-9,rtol=0) or not np.allclose(normal,ref['normal'],atol=1e-7,rtol=0):
        raise ValueError('surface point mismatch')
data.update(asset=str(asset),asset_sha256=sha256(asset),layers=layer_pins(stage),physics_contact_qualified=False)
with path.open('x') as f:json.dump(data,f,indent=2)
print('Validated surface references:',len(data['references']))
