#!/usr/bin/env python3
"""M1_B-only model, varying causal observation history and recurrent width.

The ROS 2 LOM and T-LOM generation and the spatial goal fusion are unchanged.
"""
from __future__ import annotations
import torch
from torch import nn
from m123_models import SpatialLSTMCell


class M1BMegaPolicy(nn.Module):
    def __init__(self, history_pairs=2, hidden_channels=16, img_h=100, img_w=80,
                 conv_dropouts=(.20,.15), head_dropouts=(.20,.15)):
        super().__init__()
        if history_pairs not in (1,2,3,4):
            raise ValueError('History pairs must be 1, 2, 3 or 4')
        if hidden_channels not in (12,16,24):
            raise ValueError('Hidden channels must be 12, 16 or 24')
        if img_h % 4 or img_w % 4:
            raise ValueError('Image dimensions must be divisible by four')
        if len(conv_dropouts)!=2 or len(head_dropouts)!=2 or any(not 0<=p<1 for p in (*conv_dropouts,*head_dropouts)):
            raise ValueError('Two valid convolutional and head dropout probabilities required')
        self.history_pairs = history_pairs
        self.hidden_channels = hidden_channels
        self.input_channels = history_pairs * 2
        self.img_h = img_h
        self.img_w = img_w
        self.conv_dropouts=tuple(float(p) for p in conv_dropouts)
        self.head_dropouts=tuple(float(p) for p in head_dropouts)
        self.conv1=nn.Conv2d(self.input_channels,16,kernel_size=3,stride=2,padding=1)
        self.drop1=nn.Dropout2d(self.conv_dropouts[0])
        self.conv2=nn.Conv2d(17,32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,stride=2)
        self.drop2=nn.Dropout2d(self.conv_dropouts[1])
        self.spatial_cells=nn.ModuleList([SpatialLSTMCell(32,self.hidden_channels)])
        self.flat1=nn.Linear(self.hidden_channels*(img_h//4)*(img_w//4),120)
        self.flat2=nn.Linear(120,80)
        self.out=nn.Linear(80,3)
        self.dropout_20=nn.Dropout(self.head_dropouts[0])
        self.dropout_15=nn.Dropout(self.head_dropouts[1])

    def head(self,h):
        x=self.dropout_20(torch.relu(self.flat1(torch.flatten(h,start_dim=1))))
        x=self.dropout_15(torch.relu(self.flat2(x)))
        return torch.tanh(self.out(x))

    def forward(self,x,hx=None,return_state=False,return_sequence=False):
        if x.ndim!=5 or x.shape[2:]!=(self.input_channels,self.img_h+2,self.img_w):
            raise ValueError('Input shape must be [batch,time,channels,height+2,width]')
        b,t=x.shape[:2]
        if t<1:
            raise ValueError('Empty recurrent time axis')
        if hx is None:
            cells=[None]
            prev_nn=x.new_zeros((b,3))
        else:
            if not isinstance(hx,dict) or set(hx)!={'cells','prev_nn'} or len(hx['cells'])!=1:
                raise ValueError('Invalid recurrent state structure')
            cells=list(hx['cells']);prev_nn=hx['prev_nn']
            if prev_nn.shape!=(b,3):raise ValueError('Invalid previous action')
        output_seq=[]
        for j in range(t):
            images=x[:,j,:,:self.img_h,:]
            goal=x[:,j,-1,-2:,0]
            z=self.drop1(torch.relu(self.conv1(images)))
            _,_,h,w=z.shape
            top=(h+1)//2
            gd=goal[:,0,None,None,None].expand(b,1,top,w)
            gt=goal[:,1,None,None,None].expand(b,1,h-top,w)
            z=self.drop2(self.pool(torch.relu(self.conv2(torch.cat((z,torch.cat((gd,gt),dim=2)),dim=1)))))
            cell=self.spatial_cells[0]
            hstate,cstate=cell(z,cells[0]);cells=[(hstate,cstate)]
            pred=self.head(hstate)
            output_seq.append(pred)
            prev_nn=pred.detach()
        output=torch.stack(output_seq,dim=1) if return_sequence else output_seq[-1]
        state={'cells':cells,'prev_nn':prev_nn}
        return (output,state) if return_state else output
