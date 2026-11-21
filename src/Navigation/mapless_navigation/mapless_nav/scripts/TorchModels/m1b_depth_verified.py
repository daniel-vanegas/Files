#!/usr/bin/env python3
"""M1_B + Depth using the actual M1BMegaPolicy backbone.

Keep baseline Conv1/Conv2/goal/head/state/recurrence exactly as in
m1b_mega_models.py. Add only a separate depth encoder and concatenate its
32 spatial features before the single 16-channel SpatialLSTMCell.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from m1b_mega_models import M1BMegaPolicy
from m123_models import SpatialLSTMCell


class M1BMegaDepthPolicy(M1BMegaPolicy):
    def __init__(self, history_pairs=2, hidden_channels=16,
                 img_h=100, img_w=80,
                 conv_dropouts=(.20, .15), head_dropouts=(.20, .15),
                 depth_input_hw=(480, 640), depth_resize_hw=(120, 160)):
        super().__init__(history_pairs=history_pairs,
                         hidden_channels=hidden_channels,
                         img_h=img_h, img_w=img_w,
                         conv_dropouts=conv_dropouts,
                         head_dropouts=head_dropouts)
        self.depth_input_hw = tuple(depth_input_hw)
        self.depth_resize_hw = tuple(depth_resize_hw)
        self.depth_conv1 = nn.Conv2d(history_pairs, 16, 3, stride=2, padding=1)
        self.depth_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.depth_pool = nn.AdaptiveAvgPool2d((img_h // 4, img_w // 4))
        # The baseline recurrent cell accepts 32 occupancy channels. The
        # new cell accepts 32 occupancy + 32 depth channels directly.
        self.spatial_cells = nn.ModuleList([
            SpatialLSTMCell(64, self.hidden_channels)
        ])

    @staticmethod
    def _resize_valid_depth(x, size):
        # Zero encodes invalid pixels, not an observed far-away surface.
        valid = (x > 0).to(x.dtype)
        weight = F.interpolate(valid, size=size, mode='area')
        weighted = F.interpolate(x * valid, size=size, mode='area')
        return torch.where(weight > 0, weighted / weight.clamp_min(1e-6),
                           torch.zeros_like(weighted))

    def _depth_features(self, x):
        small = self._resize_valid_depth(x, self.depth_resize_hw)
        z = torch.relu(self.depth_conv1(small))
        z = torch.relu(self.depth_conv2(z))
        return self.depth_pool(z)

    def forward(self, x, depth, hx=None, return_state=False, return_sequence=False):
        if x.ndim != 5 or x.shape[2:] != (self.input_channels, self.img_h+2, self.img_w):
            raise ValueError('Occupancy: expected [B,T,C,H+2,W] using baseline channel order')
        b,t=x.shape[:2]
        if t<1 or depth.ndim!=5 or depth.shape[:2]!=(b,t) or depth.shape[2:] != (
                self.history_pairs,*self.depth_input_hw):
            raise ValueError('Depth: expected [B,T,history_pairs,480,640], matching occupancy')
        if not torch.isfinite(x).all() or not torch.isfinite(depth).all():
            raise ValueError('Non-finite inputs')
        if torch.any(depth<0) or torch.any(depth>1):
            raise ValueError('Depth must be normalized in [0,1], zero = invalid')
        if hx is None:
            cells=[None]
            prev_nn=x.new_zeros((b,3))
        else:
            if not isinstance(hx,dict) or set(hx)!={'cells','prev_nn'} or len(hx['cells'])!=1:
                raise ValueError('Invalid baseline recurrent state')
            cells=list(hx['cells'])
            prev_nn=hx['prev_nn']
            if prev_nn.shape!=(b,3):
                raise ValueError('Invalid previous action')
        output_seq=[]
        for j in range(t):
            # This block follows M1BMegaPolicy.forward exactly.
            images=x[:,j,:,:self.img_h,:]
            goal=x[:,j,-1,-2:,0]
            z=self.drop1(torch.relu(self.conv1(images)))
            _,_,h,w=z.shape
            top=(h+1)//2
            gd=goal[:,0,None,None,None].expand(b,1,top,w)
            gt=goal[:,1,None,None,None].expand(b,1,h-top,w)
            z=self.drop2(self.pool(torch.relu(self.conv2(
                torch.cat((z,torch.cat((gd,gt),dim=2)),dim=1)))))
            zd=self._depth_features(depth[:,j])
            if z.shape[-2:] != zd.shape[-2:]:
                raise RuntimeError(f'Feature shapes differ: {z.shape} vs {zd.shape}')
            fused=torch.cat((z,zd),dim=1)
            hstate,cstate=self.spatial_cells[0](fused,cells[0])
            cells=[(hstate,cstate)]
            pred=self.head(hstate)
            output_seq.append(pred)
            prev_nn=pred.detach()
        output=torch.stack(output_seq,dim=1) if return_sequence else output_seq[-1]
        state={'cells':cells,'prev_nn':prev_nn}
        return (output,state) if return_state else output


def initialize_from_m1b(new_model: M1BMegaDepthPolicy,
                        old_model: M1BMegaPolicy) -> None:
    """Transfer exact baseline weights; leave depth-branch gate weights at zero.

    This makes the initialized Depth model numerically match the baseline in
    eval mode. Depth becomes learnable through gate gradients after updates.
    """
    if new_model.history_pairs!=old_model.history_pairs or \
       new_model.hidden_channels!=old_model.hidden_channels or \
       (new_model.img_h,new_model.img_w)!=(old_model.img_h,old_model.img_w):
        raise ValueError('Baseline/model settings do not match')
    old=old_model.state_dict()
    fresh=new_model.state_dict()
    gate_keys=[k for k in old if k.startswith('spatial_cells.0.') and old[k].ndim==4]
    if len(gate_keys)!=1:
        raise RuntimeError(f'Expected one SpatialLSTM gate tensor, got {gate_keys}')
    gate=gate_keys[0]
    c=new_model.hidden_channels
    if old[gate].shape[1]!=32+c or fresh[gate].shape[1]!=64+c:
        raise RuntimeError('Unexpected recurrent gate layout')
    for k,value in old.items():
        if k==gate:
            fresh[k].zero_()
            fresh[k][:,:32].copy_(value[:,:32])
            fresh[k][:,64:].copy_(value[:,32:])
        else:
            if k not in fresh or fresh[k].shape!=value.shape:
                raise RuntimeError(f'Incompatible inherited parameter {k}')
            fresh[k].copy_(value)
    new_model.load_state_dict(fresh,strict=True)
