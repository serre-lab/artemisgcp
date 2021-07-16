#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:13:03 2019

@author: alekhka
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as torch_models


class VGGframe(nn.Module):
    def __init__(self, num_steps, finetune=False):
        super(VGGframe, self).__init__()
        
        vggnet = torch_models.vgg19_bn(pretrained=True)
        if finetune == False:
            for param in vggnet.parameters():
                param.requires_grad = False
                
        self.features = vggnet.features
        self.avgpool = nn.AvgPool2d((5,5))
        self.classifier = nn.Linear(512, 9)

    def forward(self, xs):
        
        out = self.features(xs)
        out = self.avgpool(out)
        
        out = out.reshape(out.shape[0], -1)
        logits = self.classifier(out)
        
        
        return logits

class VGGfeats(nn.Module):
    def __init__(self, num_steps, finetune=False):
        super(VGGfeats, self).__init__()
        
        vggnet = torch_models.vgg19_bn(pretrained=True)
        if finetune == False:
            for param in vggnet.parameters():
                param.requires_grad = False
                
        self.features = vggnet.features

    def forward(self, xs):
        
        out = self.features(xs)
        
        return out

class VGGLSTM(nn.Module):
    def __init__(self, num_steps, finetune=False):
        super(VGGLSTM, self).__init__()
        self.num_steps = num_steps
        vggnet = torch_models.vgg19_bn(pretrained=True)
        if finetune == False:
            for param in vggnet.parameters():
                param.requires_grad = False
                
        self.features = vggnet.features
        self.avgpool = nn.AvgPool2d((5,5))
        
        self.lstm = nn.LSTM(512, 256)
        
        self.classifier = nn.Linear(256, 9)

    def forward(self, xs):
        
        img_feats = []
        for t in range(self.num_steps):
            x = xs[:, -self.num_steps + t, :]
            #print(t, x.shape)
            out = self.features(x)
            out = self.avgpool(out)
            out = out.reshape(out.shape[0], -1)
            
            img_feats.append(out)
            
        
        img_feats = torch.stack(img_feats)
        
        out, _ = self.lstm(img_feats)
        logits = self.classifier(out[-1]) #last timestep only
        
        
        return logits

class VGGfeatLSTM(nn.Module):
    def __init__(self, num_steps, finetune=False):
        super(VGGfeatLSTM, self).__init__()
        self.num_steps = num_steps

        #self.avgpool = nn.AvgPool2d((5,5))
        #self.conv1 = nn.Conv2d(512, 2, 1, 1)
        #self.lstm = nn.LSTM(98, 64, dropout=0.5)
        self.lstmcell = nn.LSTMCell(25088, 256)
        self.classifier = nn.Linear(256, 9)

    def forward(self, xs):
        
        #img_feats = []
        hidden, cell = torch.zeros(xs.shape[0], 256).cuda(), torch.zeros(xs.shape[0], 256).cuda()
        
        for t in range(self.num_steps):
            x = xs[:, -self.num_steps + t, :]
            #print(t, x.shape)
            #out = self.features(x)
            #out = self.avgpool(x)
            #out = self.conv1(x)
            #out = out.reshape(out.shape[0], -1)
            
            #img_feats.append(out.reshape(out.shape[0], -1))
            out = x
            hidden, cell = self.lstmcell(out.view(out.shape[0], -1), (hidden, cell))
            #hidden = F.dropout(hidden, p = 0.5)
        
        #img_feats = torch.stack(img_feats)
        
        #out, _ = self.lstm(img_feats)
        logits = self.classifier(hidden) #last timestep only
        
        
        return logits

class VGGfeatFrame(nn.Module):
    def __init__(self, num_steps, finetune=False):
        super(VGGfeatFrame, self).__init__()
        self.num_steps = num_steps

        self.avgpool = nn.AvgPool2d((7,7), stride=1, padding=0)
        #self.conv1 = nn.Conv2d(512, 2, 1, 1)
        #self.lstm = nn.LSTM(98, 64, dropout=0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, xs):
        
        #out = self.conv1(xs[:, -1, :])
        out = self.avgpool(xs[:, -1])
        out = self.fc1(out.view(xs.shape[0], -1))
        logits = self.fc2(out) #last timestep only
        
        
        return logits