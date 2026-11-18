"""Exact original CAD minimum heights, with bounded temporary tensors."""
import torch


class HeightBatch:
    def __init__(self,shapes,*,env_chunk_size=128,vertex_chunk_size=512):
        if not shapes or any(p.ndim!=2 or p.shape[1]!=3 or not len(p) for _,_,p in shapes):
            raise ValueError('nonempty original 3D vertex arrays required')
        if any(type(value) is not int or value<1
               for value in (env_chunk_size,vertex_chunk_size)):
            raise ValueError('height chunk sizes must be positive integers')
        self.env_chunk_size=env_chunk_size
        self.vertex_chunk_size=vertex_chunk_size
        device=shapes[0][2].device
        self.hands=torch.tensor([h for h,_,_ in shapes],device=device)
        self.bodies=torch.tensor([b for _,b,_ in shapes],device=device)
        width=max(len(p) for _,_,p in shapes)
        self.points=torch.zeros(len(shapes),width,3,device=device,dtype=torch.float64)
        self.valid=torch.zeros(len(shapes),width,device=device,dtype=torch.bool)
        for i,(_,_,p) in enumerate(shapes):
            self.points[i,:len(p)]=p
            self.valid[i,:len(p)]=True

    def __call__(self,poses):
        # The former N x shape x vertex matmul also materialized its broadcast
        # N x shape x 3 x vertex inputs. Bound both axes before that operation;
        # every original CAD vertex still participates in the same f64 formula.
        result=torch.empty((len(poses),len(self.hands)),device=poses.device,dtype=torch.float64)
        for env_begin in range(0,len(poses),self.env_chunk_size):
            env_end=env_begin+self.env_chunk_size
            pose=poses[env_begin:env_end,self.hands,self.bodies].double()
            w,x,y,z=pose[...,3:].unbind(-1)
            row=torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
            minimum=torch.full(pose.shape[:2],float('inf'),device=pose.device,dtype=torch.float64)
            for vertex_begin in range(0,self.points.shape[1],self.vertex_chunk_size):
                vertex_end=vertex_begin+self.vertex_chunk_size
                points=self.points[:,vertex_begin:vertex_end]
                valid=self.valid[:,vertex_begin:vertex_end]
                heights=(row.unsqueeze(-2)@points.transpose(-1,-2)).squeeze(-2)+pose[...,2,None]
                block_minimum=heights.masked_fill(~valid,float('inf')).amin(-1)
                minimum=torch.minimum(minimum,block_minimum)
            result[env_begin:env_end]=minimum
        return result
