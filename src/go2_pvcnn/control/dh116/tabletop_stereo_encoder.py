"""Scratch shared GN-ResNet18 and masked binocular cross-attention, pure torch."""
import math
import torch
from torch import nn


def position_encoding(position, width):
    """Continuous explicit sinusoidal coordinates; same device/dtype as input."""
    frequency=torch.exp(torch.arange((width+1)//2,device=position.device,dtype=position.dtype)
        *(-math.log(10000.)/max(1,(width+1)//2)))
    phase=position[...,None]*frequency
    return torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)[...,:width]


def _norm(channels):
    return nn.GroupNorm(min(32,channels//2),channels)


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,channels,3,stride,1,bias=False)
        self.norm1=_norm(channels)
        self.conv2=nn.Conv2d(channels,channels,3,1,1,bias=False)
        self.norm2=_norm(channels)
        self.downsample=nn.Identity() if in_channels==channels and stride==1 else nn.Sequential(
            nn.Conv2d(in_channels,channels,1,stride,bias=False),_norm(channels))

    def forward(self,value):
        residual=self.downsample(value)
        value=torch.relu(self.norm1(self.conv1(value)))
        return torch.relu(self.norm2(self.conv2(value))+residual)


class ResNet18(nn.Module):
    """Eight BasicBlocks, standard 64/128/256/512 widths, no pretrained weights."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),_norm(64),nn.ReLU(),nn.MaxPool2d(3,2,1))
        self.layer1=nn.Sequential(BasicBlock(64,64),BasicBlock(64,64))
        self.layer2=nn.Sequential(BasicBlock(64,128,2),BasicBlock(128,128))
        self.layer3=nn.Sequential(BasicBlock(128,256,2),BasicBlock(256,256))
        self.layer4=nn.Sequential(BasicBlock(256,512,2),BasicBlock(512,512))

    def forward(self,value):
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(value)))))


class StereoEncoder(nn.Module):
    def __init__(self,embed_dim=128,num_heads=4,image_processing_batch_size=None):
        super().__init__()
        if embed_dim<8 or embed_dim%num_heads: raise ValueError('embedding must divide attention heads')
        if image_processing_batch_size is not None and (type(image_processing_batch_size) is not int or image_processing_batch_size<1):
            raise ValueError('image processing batch size must be positive or None')
        self.embed_dim=embed_dim
        self.image_processing_batch_size=image_processing_batch_size
        self.backbone=ResNet18()
        self.project=nn.Conv2d(512,embed_dim,1)
        self.view_embedding=nn.Parameter(torch.empty(2,embed_dim))
        nn.init.normal_(self.view_embedding,std=.02)
        self.cross_attention=nn.MultiheadAttention(embed_dim,num_heads,dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(embed_dim)

    def _cross(self,query,key,query_valid,key_valid):
        mask=~key_valid[:,None].expand(-1,key.shape[1]).clone()
        mask[~key_valid,0]=False
        key=torch.where(key_valid[:,None,None],key,0.)
        context=self.cross_attention(query,key,key,key_padding_mask=mask,need_weights=False)[0]
        context=torch.where(key_valid[:,None,None],context,0.)
        return torch.where(query_valid[:,None,None],self.norm(query+context),0.)

    def forward(self,left_rgb,right_rgb,image_valid):
        if (left_rgb.ndim!=4 or right_rgb.shape!=left_rgb.shape or left_rgb.shape[1]!=3
                or image_valid.shape!=(len(left_rgb),2) or image_valid.dtype!=torch.bool):
            raise ValueError('matching RGB [B,3,H,W] and boolean view validity [B,2] required')
        if (left_rgb.dtype!=torch.float32 or right_rgb.dtype!=torch.float32
                or right_rgb.device!=left_rgb.device or image_valid.device!=left_rgb.device
                or min(left_rgb.shape[-2:])<16):
            raise ValueError('RGB float32 and view masks must share a device; H/W>=16')
        if (not bool(torch.isfinite(left_rgb).all() & torch.isfinite(right_rgb).all())
                or bool(((left_rgb<0)|(left_rgb>1)).any()|((right_rgb<0)|(right_rgb>1)).any())):
            raise ValueError('RGB must be finite in [0,1]')
        left=torch.where(image_valid[:,0,None,None,None],left_rgb,0.)
        right=torch.where(image_valid[:,1,None,None,None],right_rgb,0.)
        images=torch.cat((left,right),0)
        batch=self.image_processing_batch_size
        if batch is None:
            features=self.project(self.backbone(images))
        else:
            chunks=[]
            for start in range(0,len(images),batch):
                image=images[start:start+batch]
                if self.training and torch.is_grad_enabled():
                    # GN and dropout0 are deterministic. Checkpointing avoids
                    # retaining every high-resolution ResNet activation while
                    # preserving parameter/input gradients across image chunks.
                    from torch.utils.checkpoint import checkpoint
                    output=checkpoint(lambda value:self.project(self.backbone(value)),image,use_reentrant=False)
                else:output=self.project(self.backbone(image))
                chunks.append(output)
            features=torch.cat(chunks,0)
        n=left.shape[0]; h,w=features.shape[-2:]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=left.device),
            torch.linspace(-1,1,w,device=left.device),indexing='ij')
        position=position_encoding(xx.flatten(),self.embed_dim//2)
        position=torch.cat((position,position_encoding(yy.flatten(),self.embed_dim-self.embed_dim//2)),-1)
        tokens=features.flatten(2).transpose(1,2)+position[None]
        l,r=tokens[:n]+self.view_embedding[0],tokens[n:]+self.view_embedding[1]
        lout=self._cross(l,r,image_valid[:,0],image_valid[:,1])
        rout=self._cross(r,l,image_valid[:,1],image_valid[:,0])
        valid=image_valid[:,:,None].expand(n,2,h*w).flatten(1)
        return torch.cat((lout,rout),1),valid
