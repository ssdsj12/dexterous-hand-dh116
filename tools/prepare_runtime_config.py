"""Revalidate relocated geometry and generate NEW runtime configuration pins."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd
from tools.migration_copy import sha256,copy_file
from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene
from go2_pvcnn.control.dh116.usd_surface import read_surface
from go2_pvcnn.control.dh116.contact_frames import surface_reference
from go2_pvcnn.control.dh116.path_certificate import certify_joint_path
from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
from go2_pvcnn.control.dh116.approach import load_approach
from go2_pvcnn.control.dh116.pose_artifact import load_active_pose

root=Path(__file__).resolve().parents[1]
old=root/'archive/dh116_v2';assetdir=root/'assets/fixed_dual_panda_dh116_v1'
asset=assetdir/'fixed_dual_panda_dh116.usd';scene=SourceCollisionScene(Usd.Stage.Open(str(asset)))
out=root/'configs/table_margin_raise1mm'
if out.exists():raise FileExistsError(out)
relocation=json.loads((assetdir/'relocation.json').read_text())
mapping={x['source']:assetdir/x['destination'] for x in relocation['layers']}
manifest=json.loads((root/'archive/fixed_dual_panda_dh116_v1/manifest.json').read_text())
manifest['asset_sha256']=sha256(asset)
for s in manifest['sources']:
    s['path']=str(mapping[s['path']]);s['sha256']=sha256(Path(s['path']))
manifest['physics_verified']=False
manifest_path=assetdir/'manifest.json'
if manifest_path.exists():
    if json.loads(manifest_path.read_text())!=manifest:raise ValueError('conflicting relocated manifest')
else:
    with manifest_path.open('x') as f:json.dump(manifest,f,indent=2)
refs=json.loads((old/'postpilot7_point_references_v1.json').read_text())
stage=Usd.Stage.Open(str(asset))
for ref in refs['references']:
    model=scene.models[ref['side']]
    body=ref.get('body_name',{'thumb':'finger14_link','index':'finger23_link'}[ref['digit']])
    surface=read_surface(stage,model.bodies[body])
    triangle=np.asarray(surface['points_body'])[np.asarray(surface['triangles'])[ref['triangle_index']]]
    point,normal=surface_reference(triangle,ref['weights'])
    if not np.allclose(point,ref['point'],atol=1e-9,rtol=0) or not np.allclose(normal,ref['normal'],atol=1e-7,rtol=0):
        raise ValueError('reference surface coordinates changed')
refs['asset_sha256']=sha256(asset);refs['layers']=scene.layer_pins
refs['scope']='Relocated source geometry validated; physical grasp NOT qualified.'
out.mkdir(parents=True)
def write(name,data):
    with (out/name).open('x') as f:json.dump(data,f,indent=2)
write('point_references.json',refs)
contact=json.loads((old/'contact_hulls_v2/manifest.json').read_text())
for row in contact['meshes'].values():
    if read_surface(stage,row['body_path'])['geometry_sha256']!=row['geometry_sha256']:
        raise ValueError('contact source geometry changed')
    copy_file(old/'contact_hulls_v2'/row['path'],out,Path('contact')/row['path'],'contact_mesh')
contact['asset_sha256']=sha256(asset);contact['layers']=scene.layer_pins
write('contact/manifest.json',contact)
pose=json.loads((old/'table_margin_raise1mm_20261018/prepose.json').read_text())
pose.update(asset_sha256=sha256(asset),layers=scene.layer_pins,grasp_qualified=False)
original=json.loads((old/'table_margin_raise1mm_20261018/approach.json').read_text())
wire=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.308)]
report=certify_joint_path(original['input']['waypoints'],lambda q,d,leg:scene.check(q,wire,active_half_delta=d))
if not report['static_wire_source_sweep_passed']:raise ValueError('relocated path rejected')
initial=wire.copy();initial[:,2]=.318
if not scene.check(pose['states'],initial)['source_clearance_passed']:raise ValueError('initial pose rejected')
report.update(layers=scene.layer_pins,input=original['input'],grasp_qualified=False)
write('prepose.json',pose);write('approach.json',report)
load_active_pose(out/'prepose.json',sha256(out/'prepose.json'),sha256(asset),scene.layer_pins)
load_approach(out/'approach.json',sha256(out/'approach.json'),scene.layer_pins)
load_mesh_manifest(out/'contact/manifest.json',sha256(out/'contact/manifest.json'))
write('validation.json',dict(source_geometry_validated=True,path_checks=report['checks'],
    physics_validated=False,old_stage_states_reusable=False,
    files={name:sha256(out/name) for name in ('prepose.json','approach.json','point_references.json','contact/manifest.json')}))
print('Runtime source configuration validated:',out,'checks:',report['checks'],flush=True)
