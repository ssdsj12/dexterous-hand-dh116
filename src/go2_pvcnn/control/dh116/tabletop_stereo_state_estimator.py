"""Stereo-only spatial/temporal estimation and canonical24 latent Actor."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .tabletop_stereo_encoder import StereoEncoder, position_encoding
from .tabletop_stereo_observation import StereoHistoryBatch
from .tabletop_stereo_temporal import StereoTemporal
from .tabletop_vision_command_shadow import action_contract_sha256 as canonical_action_sha256


# force_proxy: per-hand thumb/index/middle accepted-pad normal proxies in N;
# not world XYZ forces and not deployed tactile measurements.
HEAD_SHAPES = dict(privileged_state=(371,),geometry=(128,3),velocity=(128,3),
    deformation=(128,),contact_logits=(2,3),force_proxy=(2,3),slip_logits=(2,),slip_velocity=(2,))


class StereoStateEstimator(nn.Module):
    def __init__(self,embed_dim=128,history=4,num_heads=4,num_layers=2,image_processing_batch_size=None):
        super().__init__()
        self.embed_dim,self.history=embed_dim,history
        self.encoder=StereoEncoder(embed_dim,num_heads,image_processing_batch_size)
        self.depth_project=nn.Linear(5,embed_dim)
        self.point_project=nn.Linear(3,embed_dim)
        self.proprio_project=nn.Sequential(nn.Linear(64,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim))
        # Frame identifiers distinguish world geometry, image geometry and
        # measured joint/FK state; no simulator truth is a runtime argument.
        self.frame_embedding=nn.Parameter(torch.empty(3,embed_dim))
        nn.init.normal_(self.frame_embedding,std=.02)
        self.null_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.spatial_attention=nn.MultiheadAttention(embed_dim,num_heads,dropout=0.,batch_first=True)
        self.gate=nn.Linear(2*embed_dim,embed_dim)
        self.spatial_norm=nn.LayerNorm(embed_dim)
        self.temporal=StereoTemporal(embed_dim,history,num_heads,num_layers)
        self.heads=nn.ModuleDict({name:nn.Linear(embed_dim,math.prod(shape)) for name,shape in HEAD_SHAPES.items()})

    def _spatial(self,h):
        b,t=h.history_valid.shape;n=b*t;d=self.embed_dim
        active=h.history_valid.flatten()
        views=(h.image_valid & h.history_valid[:,:,None]).flatten(0,1)
        left=torch.where(h.history_valid[:,:,None,None,None],h.left_rgb,0.).flatten(0,1)
        right=torch.where(h.history_valid[:,:,None,None,None],h.right_rgb,0.).flatten(0,1)
        stereo,stereo_valid=self.encoder(left,right,views)
        pair=views.all(1)
        depth_valid=(h.depth_valid & h.history_valid[:,:,None,None]).flatten(0,1)&pair[:,None,None]
        depth=torch.where(depth_valid,h.depth_m.flatten(0,1),0.)
        confidence=torch.where(depth_valid,h.depth_confidence.flatten(0,1),0.)
        # Mask-weighted pooling never turns a missing depth pixel into a label.
        weight=F.adaptive_avg_pool2d(depth_valid.float()[:,None],(3,3))
        pooled=F.adaptive_avg_pool2d(depth[:,None],(3,3))/weight.clamp_min(1e-8)
        conf=F.adaptive_avg_pool2d(confidence[:,None],(3,3))/weight.clamp_min(1e-8)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,3,device=left.device),
            torch.linspace(-1,1,3,device=left.device),indexing='ij')
        coordinates=torch.stack((xx.flatten(),yy.flatten()),-1)[None].expand(n,-1,-1)
        depth_input=torch.cat((pooled.flatten(2).transpose(1,2),conf.flatten(2).transpose(1,2),
            weight.flatten(2).transpose(1,2),coordinates),-1)
        depth_tokens=self.depth_project(depth_input)+self.frame_embedding[0]
        depth_mask=weight.flatten(1)>0
        point_mask=(h.point_valid & h.history_valid[:,:,None]).flatten(0,1)&pair[:,None]
        points=torch.where(point_mask[:,:,None],h.points_world.flatten(0,1),0.)
        point_tokens=self.point_project(points)+self.frame_embedding[1]
        point_tokens=point_tokens+position_encoding(torch.linspace(0,1,128,device=left.device),d)[None]
        # Point ordinal IDs preserve upstream arc-length identity; they do not
        # infer topology or claim cross-frame correspondence of an unordered set.
        proprio=torch.where(active[:,None],h.proprio.flatten(0,1),0.)
        timing=torch.stack((torch.where(active,h.dt.flatten(),0.),
                            torch.where(active,h.image_age.flatten(),0.)),-1)
        query=self.proprio_project(torch.cat((proprio,timing),-1))+self.frame_embedding[2]
        tokens=torch.cat((stereo,depth_tokens,point_tokens,self.null_token.expand(n,-1,-1)),1)
        valid=torch.cat((stereo_valid,depth_mask,point_mask,torch.ones(n,1,device=left.device,dtype=torch.bool)),1)
        tokens=torch.where(valid[:,:,None],tokens,0.)
        visual=self.spatial_attention(query[:,None],tokens,tokens,key_padding_mask=~valid,need_weights=False)[0][:,0]
        gated=torch.sigmoid(self.gate(torch.cat((query,visual),-1)))*visual
        return torch.where(active[:,None],self.spatial_norm(query+gated),0.).reshape(b,t,d)

    def forward(self,history):
        if type(history) is not StereoHistoryBatch:
            raise TypeError('Actor estimator accepts only StereoHistoryBatch; privileged mappings forbidden')
        history.validate()
        if history.history_valid.shape[1]!=self.history: raise ValueError('history length mismatch')
        frames=self._spatial(history)
        latent=self.temporal(frames,history.history_valid,history.dt,history.image_age)
        result={'latent':latent}
        result.update({name:head(latent).reshape(len(history),*HEAD_SHAPES[name]) for name,head in self.heads.items()})
        return result


class StereoStudentActor(nn.Module):
    """Unbounded mu24 and latent std; use existing saved-latent Gaussian math."""
    def __init__(self,estimator=None,*,embed_dim=128,action_contract_sha256=None,image_processing_batch_size=None):
        super().__init__()
        expected=canonical_action_sha256()
        if action_contract_sha256 is not None and action_contract_sha256!=expected:
            raise ValueError('canonical action contract SHA256 mismatch')
        self.action_contract_sha256=expected
        self.estimator=StereoStateEstimator(embed_dim=embed_dim,image_processing_batch_size=image_processing_batch_size) if estimator is None else estimator
        if not isinstance(self.estimator,StereoStateEstimator): raise TypeError('StereoStateEstimator required')
        self.action_net=nn.Sequential(nn.Linear(self.estimator.embed_dim,128),nn.Tanh(),nn.Linear(128,24))
        nn.init.orthogonal_(self.action_net[-1].weight,.01);nn.init.zeros_(self.action_net[-1].bias)
        self.log_std=nn.Parameter(torch.full((24,),math.log(.1)))

    def latent_mean(self,history):
        return self.action_net(self.estimator(history)['latent'])

    def forward(self,history):
        return self.latent_mean(history)

    def deterministic(self,history):
        return self.latent_mean(history).tanh()
