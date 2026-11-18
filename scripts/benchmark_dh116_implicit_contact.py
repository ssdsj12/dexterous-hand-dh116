"""Short numerical gate for the opt-in implicit backend (not Isaac grasp proof)."""
from __future__ import annotations
import argparse, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import open3d as o3d

from co_sim.moving_contact import MeshAsset, MovingContactRodEngine
from co_sim.implicit_moving_contact import ImplicitMovingContactRodEngine
from co_sim.moving_contact_models import MeshMotion
from co_sim.models import CoSimConfig

CHANNELS=("left_thumb","left_index","right_thumb","right_index")

def config(dt):
    return CoSimConfig(py_dt=dt, isaac_dt=1e-3, n_elem=4, base_length=.2,
        base_radius=.008, density=1000., youngs_modulus=1e5,
        damping_constant=0., rod_start=np.array([-.1,0.,.2]),
        rod_direction=np.array([1.,0.,0.]), rod_normal=np.array([0.,0.,1.]))

def motion(pos, vel=(0.,0.,0.)):
    return MeshMotion(position_w=np.asarray(pos,float), rotation_wb=np.eye(3),
        velocity_w=np.asarray(vel,float), omega_w=np.zeros(3),
        acceleration_w=np.zeros(3), alpha_w=np.zeros(3))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--duration',type=float,default=.001)
    p.add_argument('--py-dt',type=float,default=2.5e-6); p.add_argument('--moving',action='store_true')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'box.ply'; mesh=o3d.geometry.TriangleMesh.create_box(.02,.02,.02)
        mesh.translate([-.01,-.01,-.01]); mesh.compute_triangle_normals(); assert o3d.io.write_triangle_mesh(str(path),mesh,write_ascii=True)
        asset=MeshAsset(path,hashlib.sha256(path.read_bytes()).hexdigest()); assets={c:asset for c in CHANNELS}
        kw=dict(table_center=(0,0,-1),table_size=(2,2,.1),meshes=assets,contact_subsamples=4,velocity_damping_coefficient=1000.)
        ref=MovingContactRodEngine(config(a.py_dt),**kw); cand=ImplicitMovingContactRodEngine(config(a.py_dt),**kw)
        motions={c:motion((10+i,10,10),(.01,0,0) if a.moving else (0,0,0)) for i,c in enumerate(CHANNELS)}
        motions['left_thumb']=motion((0,0,.183),(.01,0,0) if a.moving else (0,0,0))
        steps=int(round(a.duration/a.py_dt)); maxpos=maxforce=maximp=0.; balance=0.; ref_peak=0.; cand_peak=0.; ref_imp=np.zeros(3); cand_imp=np.zeros(3)
        for _ in range(steps):
            sr,ir=ref.advance_with_meshes(motions,a.py_dt); sc,ic=cand.advance_with_meshes(motions,a.py_dt)
            maxpos=max(maxpos,float(np.max(np.abs(sr['rod_position']-sc['rod_position']))))
            for c in CHANNELS:
                ref_peak=max(ref_peak,ir[c].peak_force_n); cand_peak=max(cand_peak,ic[c].peak_force_n)
                ref_imp+=ir[c].linear_w; cand_imp+=ic[c].linear_w
                balance=max(balance,float(np.linalg.norm(cand.last_contact_diagnostics['channels'][c]['linear_balance_w'])))
        maxforce=abs(cand_peak-ref_peak); maximp=float(np.linalg.norm(cand_imp-ref_imp))
    result={'duration_s':a.duration,'py_dt_s':a.py_dt,'moving_boundary':a.moving,'steps':steps,
        'max_position_difference_m':maxpos,'peak_force_difference_n':maxforce,
        'linear_impulse_difference_ns':maximp,'max_linear_balance_ns':balance,
        'reference_peak_n':ref_peak,'candidate_peak_n':cand_peak,
        'candidate_final':{k:np.asarray(sc[k]).tolist() for k in ('rod_position','rod_velocity','rod_director','rod_omega')},
        'criteria':{'position_m':bool(maxpos<=2e-4),'force_n':bool(maxforce<=max(.05,.05*max(ref_peak,1e-12))),
                    'impulse_ns':bool(maximp<=max(1e-4,.05*max(np.linalg.norm(ref_imp),1e-12))),
                    'balance_ns':bool(balance<=1e-9)}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); raise SystemExit(0 if all(result['criteria'].values()) else 2)
if __name__=='__main__': main()
