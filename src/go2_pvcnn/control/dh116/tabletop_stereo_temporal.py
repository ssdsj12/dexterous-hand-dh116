"""Four-frame temporal Transformer with explicit index and measured time codes."""
import torch
from torch import nn
from .tabletop_stereo_encoder import position_encoding


class StereoTemporal(nn.Module):
    def __init__(self,embed_dim=128,history=4,num_heads=4,num_layers=2):
        super().__init__()
        if history<1 or num_layers<1 or embed_dim%num_heads: raise ValueError('invalid temporal dimensions')
        self.embed_dim,self.history=embed_dim,history
        layer=nn.TransformerEncoderLayer(embed_dim,num_heads,4*embed_dim,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers,enable_nested_tensor=False)
        self.time_project=nn.Linear(2,embed_dim)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,frames,history_valid,dt=None,image_age=None):
        if (frames.ndim!=3 or frames.shape[1:]!=(self.history,self.embed_dim)
                or history_valid.shape!=frames.shape[:2] or history_valid.dtype!=torch.bool
                or frames.dtype!=torch.float32 or history_valid.device!=frames.device):
            raise ValueError('temporal frames [B,T,D] and boolean history mask required')
        if not bool(torch.isfinite(frames).all()): raise ValueError('finite frame embeddings required')
        frames=torch.where(history_valid[:,:,None],frames,0.)
        position=position_encoding(torch.arange(self.history,device=frames.device,dtype=frames.dtype),self.embed_dim)
        explicit_dt=dt is not None
        if dt is None: dt=torch.zeros_like(history_valid,dtype=frames.dtype)
        if image_age is None: image_age=torch.zeros_like(dt)
        if (dt.shape!=history_valid.shape or image_age.shape!=dt.shape
                or dt.device!=frames.device or image_age.device!=frames.device
                or dt.dtype!=frames.dtype or image_age.dtype!=frames.dtype):
            raise ValueError('time arrays must have shape [B,T]')
        if (not bool(torch.isfinite(dt).all() & torch.isfinite(image_age).all())
                or bool((image_age<0).any()) or (explicit_dt and bool(((dt<=0)&history_valid).any()))):
            raise ValueError('finite positive active dt and nonnegative image age required')
        time=torch.stack((torch.where(history_valid,dt,0.).cumsum(1),
            torch.where(history_valid,image_age,0.)),-1)
        frames=frames+position[None]+self.time_project(time)
        safe=history_valid.clone(); empty=~safe.any(1);safe[empty,0]=True
        encoded=self.transformer(frames,src_key_padding_mask=~safe)
        index=(history_valid.long()*torch.arange(1,self.history+1,device=frames.device)).amax(1)-1
        selected=encoded[torch.arange(len(frames),device=frames.device),index.clamp_min(0)]
        return torch.where(empty[:,None],torch.zeros_like(selected),self.norm(selected))
