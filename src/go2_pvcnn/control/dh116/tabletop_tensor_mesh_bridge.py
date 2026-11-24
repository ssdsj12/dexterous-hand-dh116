"""One device pose submission to the original CUDA mesh ABI; no stepping."""
from copy import deepcopy
import math
import torch

class CudaTensorView:
    """Hold a live default-stream torch tensor for the existing CAI binding."""
    def __init__(self,tensor):
        if (not isinstance(tensor,torch.Tensor) or tensor.device.type!="cuda"
                or tensor.dtype!=torch.float32 or not tensor.is_contiguous()):
            raise ValueError("contiguous CUDA float32 target buffer required")
        if torch.cuda.current_stream(tensor.device)!=torch.cuda.default_stream(tensor.device):
            raise ValueError("existing solver ABI requires original legacy/default CUDA stream")
        self.tensor=tensor
        self.__cuda_array_interface__=dict(shape=tuple(tensor.shape),strides=None,
            typestr="<f4",data=(tensor.data_ptr(),False),version=3,stream=1)

class TensorMeshBridge:
    def __init__(self,mesh_names,bindings,initial_positions,initial_quaternions):
        if (initial_positions.ndim!=3 or initial_positions.shape[1:]!=(len(mesh_names),3)
                or initial_quaternions.shape!=(*initial_positions.shape[:2],4)
                or initial_positions.device!=initial_quaternions.device
                or len(set(mesh_names))!=len(mesh_names)):
            raise ValueError("complete same-device initial mesh layout required")
        if not all(bool(torch.isfinite(t).all()) for t in (initial_positions,initial_quaternions)):
            raise ValueError("finite initial mesh poses required")
        if not torch.allclose(initial_quaternions.norm(dim=-1),torch.ones_like(initial_quaternions[...,0]),atol=1e-5,rtol=0):
            raise ValueError("unit initial XYZW mesh quaternions required")
        self.names=tuple(mesh_names);self.n=len(initial_positions)
        self.positions=initial_positions.detach().float().clone().contiguous()
        self.quaternions=initial_quaternions.detach().float().clone().contiguous()
        slots=[];sides=[];links=[]
        for name,row in bindings.items():
            if name not in self.names or row["side"] not in ("left","right") or not 0<=row["body_index"]<23:
                raise ValueError("actual original contact link binding required")
            slots.append(self.names.index(name));sides.append(0 if row["side"]=="left" else 1);links.append(row["body_index"])
        dev=self.positions.device
        self.slots=torch.tensor(slots,device=dev)
        self.sides=torch.tensor(sides,device=dev)
        self.links=torch.tensor(links,device=dev)
        self._material=self._resets=None;self._views=None

    @torch.no_grad()
    def targets(self,link_poses):
        if (not isinstance(link_poses,torch.Tensor) or link_poses.shape!=(self.n,2,23,7)
                or link_poses.device!=self.positions.device or not link_poses.is_floating_point()
                or not bool(torch.isfinite(link_poses).all())
                or not torch.allclose(link_poses[...,3:].norm(dim=-1),
                    torch.ones_like(link_poses[...,0]),atol=1e-5,rtol=0)):
            raise ValueError("complete finite unit same-device actual link poses required")
        selected=link_poses[:,self.sides,self.links]
        self.positions[:,self.slots]=selected[...,:3]
        self.quaternions[:,self.slots]=selected[...,3:]
        return self.positions,self.quaternions

    def submit(self,solver,link_poses):
        if solver.num_envs!=self.n or solver.last_error!=0:
            raise ValueError("same healthy actual shared solver required")
        time=float(solver.time)
        p,q=self.targets(link_poses)
        self._views=(CudaTensorView(p),CudaTensorView(q))
        solver.set_mesh_poses_all(*self._views,teleport=False)
        if solver.last_error!=0 or float(solver.time)!=time:
            raise ValueError("pose target submission changed actual solver boundary")

    def bind_material(self,parameters,*,reset_ids):
        """Startup/Reset verifier must supply its already-qualified actual snapshot."""
        if (len(reset_ids)!=self.n or any(type(i)is not int or i<1 for i in reset_ids)
                or not parameters or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in parameters.values())):
            raise ValueError("positive actual material snapshot and complete Reset IDs required")
        if self._material is not None and parameters!=self._material:
            raise ValueError("frozen material changed without independent task requalification")
        self._material=deepcopy(parameters);self._resets=tuple(reset_ids)

    def material(self,*,reset_ids):
        if self._material is None or tuple(reset_ids)!=self._resets:
            raise ValueError("startup or selected Reset material qualification required")
        return self._material
