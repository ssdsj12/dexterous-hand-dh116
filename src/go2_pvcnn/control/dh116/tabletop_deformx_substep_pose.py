"""CPU reconstruction of locked SDF mesh interpolation, never measured contact.

This mirrors csrc/sdf.cu stageMotion / iteration.cu initItr. NumPy
float32 does not guarantee CUDA FMA/trigonometric bit equivalence. Accordingly
neither positive nor negative effective contact is qualified by this module.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

VERSION="dh116-source-derived-substep-pose-v1"
SUBSTEPS=12
SOURCE_LOCK={
"csrc/bindings.cpp":"1be4aad4bfc6de1c4a2c44e891d57ce7e163b78c24a8bf99619d1cb1fe0ce836",
"csrc/sdf.cu":"e5b3b1b00881aa1aff1697214f36b4e12310f8d62655455be7ce6b6df0369c12",
"third_party/yarnball/KittenEngine/YarnBall/sim/iteration.cu":"84dab66f671bc0052640b55a89056acbc319071a42f4d9197ff8199acad8753c",
"third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh":"8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb",
}
BACKEND_SHA="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"

def verify_source_lock(root,lock=SOURCE_LOCK):
    root=Path(root).resolve(strict=True)
    if not lock:raise ValueError("nonempty source lock required")
    for name,want in lock.items():
        path=root/name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source path escape")
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("regular source within root required")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:
            raise ValueError("source SHA differs: "+name)
    return dict(lock)

def target_pose(value):
    p=np.asarray(value,dtype=float)
    if (p.shape!=(7,) or not np.isfinite(p).all()
        or abs(np.linalg.norm(p[3:])-1)>1e-5):
        raise ValueError("finite world xyz and unit XYZW pose required")
    p=p.astype(np.float32,copy=True)
    p[3:]/=np.sqrt(np.dot(p[3:],p[3:]))
    return p

def _mul(a,b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),
                 a[3]*b[3]-np.dot(a[:3],b[:3])].astype(np.float32)

def pose_at_substep(start,end,substep):
    if type(substep) is not int or not 0<=substep<SUBSTEPS:
        raise ValueError("zero-based integer solver substep 0..11 required")
    p0,p1=target_pose(start),target_pose(end)
    q0,q1=p0[3:],p1[3:]
    relative=_mul(q1,np.r_[-q0[:3],q0[3]])
    if relative[3]<0:relative=-relative
    s=np.sqrt(np.dot(relative[:3],relative[:3]))
    rot=np.zeros(3,dtype=np.float32) if not s>1e-9 else (
        relative[:3]*(np.float32(2)*np.arctan2(s,relative[3])/s))
    alpha=np.float32(substep+1)/np.float32(SUBSTEPS)
    angle=np.sqrt(np.dot(rot,rot))
    if angle>0:
        half=np.float32(.5)*alpha*angle
        dq=np.r_[(rot/angle)*np.sin(half),np.cos(half)].astype(np.float32)
        q=_mul(dq,q0);q/=np.sqrt(np.dot(q,q))
    else:q=q0.copy()
    return np.r_[p0[:3]+alpha*(p1[:3]-p0[:3]),q].astype(np.float32)

def to_matrix(pose):
    p=target_pose(pose);out=np.eye(4)
    out[:3,:3]=Rotation.from_quat(p[3:]).as_matrix();out[:3,3]=p[:3]
    return out

class DeferredMeshPose:
    """Testable source protocol, not a solver bridge or native measurement."""
    def __init__(self,initial):
        # add_mesh commits the initial target; set_mesh_poses does not.
        self.current=target_pose(initial);self.target=self.current.copy();self.teleport=False
    def set_target(self,pose,*,teleport):
        if type(teleport) is not bool:raise ValueError("explicit teleport flag")
        self.target=target_pose(pose);self.teleport=teleport
    def step_poses(self):
        start=self.target if self.teleport else self.current
        poses=[pose_at_substep(start,self.target,k) for k in range(SUBSTEPS)]
        self.current=self.target.copy();self.teleport=False
        return poses

def classification_status(candidate):
    if type(candidate) is not bool:raise ValueError("explicit CPU candidate")
    return dict(cpu_source_derived_candidate=candidate,effective_contact_value=None,
        native_substep_pose_measured=False,cuda_bitwise_equivalence_verified=False,
        positive_contact_qualified=False,negative_contact_qualified=False,
        contact_depth_qualified=False,teacher_admission=False)
