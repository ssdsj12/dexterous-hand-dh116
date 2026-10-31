"""Optional exact batching of original CAD vertex minimum heights."""
import torch


class HeightBatch:
    def __init__(self,shapes):
        if not shapes or any(p.ndim!=2 or p.shape[1]!=3 or not len(p) for _,_,p in shapes):
            raise ValueError('nonempty original 3D vertex arrays required')
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
        pose=poses[:,self.hands,self.bodies].double()
        w,x,y,z=pose[...,3:].unbind(-1)
        row=torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
        heights=(row.unsqueeze(-2)@self.points.transpose(-1,-2)).squeeze(-2)+pose[...,2,None]
        return heights.masked_fill(~self.valid,float('inf')).amin(-1)
