#!/usr/bin/env python3
"""M1/M2/M3 goal-conditioned mapless ConvLSTM policies.

The robot receives no global pose, map, route or external obstacle memory.
Each recurrent step uses two occupancy channels and the current goal.
"""
from __future__ import annotations
import torch
from torch import nn


class SpatialLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)
        with torch.no_grad():
            self.gates.bias[hidden_channels:2*hidden_channels].fill_(1)

    def forward(self, x, state=None):
        if state is None:
            size = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
            h = x.new_zeros(size)
            c = x.new_zeros(size)
        else:
            h,c = state
        i,f,o,g = self.gates(torch.cat((x,h),dim=1)).chunk(4,dim=1)
        c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h = torch.sigmoid(o)*torch.tanh(c)
        return h,c


class M123Policy(nn.Module):
    """M1, M2 or M3 without flattening before the recurrent layers.

    Inputs [B,T,2,H+2,W]. Last 2 rows of current channel hold (d,theta).
    At every recurrent step the two image channels are:
      M1: [LOM(t), T-LOM(t)]
      M2/M3: [T-LOM(t-1), T-LOM(t)]
    """
    SPECS = {
        'M1': {'filters':(16,32),'dropouts':(0.20,0.15),'layers':1,'bptt':(16,),'fusion':'rows'},
        'M2': {'filters':(8,16),'dropouts':(0.15,0.10),'layers':1,'bptt':(16,),'fusion':'columns'},
        'M3': {'filters':(8,16),'dropouts':(0.15,0.10),'layers':2,'bptt':(16,8),'fusion':'columns'},
    }

    def __init__(self, variant: str, img_h: int = 100, img_w: int = 80,
                 hidden_channels: int = 16, head_dropouts=(0.20,0.15)):
        super().__init__()
        variant=variant.upper()
        if variant not in self.SPECS:
            raise ValueError(f'Unknown M123 variant {variant}')
        if min(img_h,img_w,hidden_channels)<4 or img_h%4 or img_w%4:
            raise ValueError('Grid dimensions must be >=4 and divisible by 4')
        self.variant=variant
        self.img_h=int(img_h)
        self.img_w=int(img_w)
        self.hidden_channels=int(hidden_channels)
        self.spec=self.SPECS[variant]
        n1,n2=self.spec['filters']
        p1,p2=self.spec['dropouts']
        # Conv1 stride 2 followed by goal fusion at intermediate resolution.
        self.conv1=nn.Conv2d(2,n1,kernel_size=3,stride=2,padding=1)
        self.drop1=nn.Dropout2d(p=p1)
        self.conv2=nn.Conv2d(n1+1,n2,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop2=nn.Dropout2d(p=p2)
        self.spatial_cells=nn.ModuleList([
            SpatialLSTMCell(n2 if i==0 else hidden_channels,hidden_channels)
            for i in range(self.spec['layers'])
        ])
        reduced_h,reduced_w=self.img_h//4,self.img_w//4
        # Exactly three dense layers, as in the historical RNN output head.
        self.flat1=nn.Linear(hidden_channels*reduced_h*reduced_w,120)
        self.flat2=nn.Linear(120,80)
        self.out=nn.Linear(80,3)
        self.dropout_20=nn.Dropout(p=float(head_dropouts[0]))
        self.dropout_15=nn.Dropout(p=float(head_dropouts[1]))

    @staticmethod
    def mixed_target(target: torch.Tensor, height: int, width: int, mode: str):
        """Return one mixed goal map; target [...,2] contains d and theta."""
        if target.shape[-1]!=2 or height<2 or width<2:
            raise ValueError('Target must contain d, theta and map >=2x2')
        if mode=='rows':
            top=(height+1)//2
            d=target[...,0,None,None,None].expand(*target.shape[:-1],1,top,width)
            th=target[...,1,None,None,None].expand(*target.shape[:-1],1,height-top,width)
            return torch.cat((d,th),dim=-2)
        if mode=='columns':
            # Even columns: d. Odd columns: theta.
            mask=(torch.arange(width,device=target.device)%2==0).view(*([1]*(target.ndim-1)),1,1,width)
            d=target[...,0,None,None,None]
            th=target[...,1,None,None,None]
            return torch.where(mask,d,th).expand(*target.shape[:-1],1,height,width)
        raise ValueError('Invalid goal fusion')

    def head(self,h):
        x=torch.flatten(h,start_dim=1)
        x=self.dropout_20(torch.relu(self.flat1(x)))
        x=self.dropout_15(torch.relu(self.flat2(x)))
        return torch.tanh(self.out(x))

    def forward(self,x:torch.Tensor,hx=None,return_state=False,return_sequence=False):
        if x.ndim!=5 or tuple(x.shape[2:])!=(2,self.img_h+2,self.img_w):
            raise ValueError(f'Expected [B,T,2,{self.img_h+2},{self.img_w}], got {tuple(x.shape)}')
        batch,steps=x.shape[:2]
        if steps<1:
            raise ValueError('Empty temporal sequence')
        if hx is not None and len(hx)!=len(self.spatial_cells):
            raise ValueError('Incorrect number of ConvLSTM states')
        # Only input current goal; paired historical target rows are not used.
        goal=x[:,:,1,-2:,0]
        occupancy=x[:,:,:,:self.img_h,:].reshape(batch*steps,2,self.img_h,self.img_w)
        stage1=self.drop1(torch.relu(self.conv1(occupancy)))
        _,_,h,w=stage1.shape
        target_map=self.mixed_target(goal,h,w,self.spec['fusion']).reshape(batch*steps,1,h,w)
        stage2=self.drop2(self.pool(torch.relu(self.conv2(torch.cat((stage1,target_map),dim=1)))))
        _,feature_channels,fh,fw=stage2.shape
        stage2=stage2.reshape(batch,steps,feature_channels,fh,fw)
        states=list(hx) if hx is not None else [None]*len(self.spatial_cells)
        predictions=[]
        for step in range(steps):
            z=stage2[:,step]
            new_states=[]
            for layer,cell in enumerate(self.spatial_cells):
                old=states[layer]
                # M3: layer 1 propagates gradients through 16 steps; layer 2
                # only through 8. Values persist unchanged across detach.
                if (old is not None and self.training and torch.is_grad_enabled()
                        and layer>0 and step>0 and step%self.spec['bptt'][layer]==0):
                    old=(old[0].detach(),old[1].detach())
                h,c=cell(z,old)
                z=h
                new_states.append((h,c))
            states=new_states
            if return_sequence:
                predictions.append(self.head(z))
        result=torch.stack(predictions,dim=1) if return_sequence else self.head(z)
        return (result,states) if return_state else result
