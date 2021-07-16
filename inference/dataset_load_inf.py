#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:49:24 2019

@author: alekhka
"""
#import numpy as np
import pickle
#import os
import torch
from torch.utils.data.dataset import Dataset


class MouseDataset(Dataset):
    def __init__(self, data_path):
        self.frame_steps = 64
        self.frame_stride = 1
        with open(data_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            vid_embs = u.load()
            #vid_embs = pickle.load(f)

        self.vid_embs = vid_embs
        self.tot_frames = len(vid_embs) - 64
        print(data_path, max(list(vid_embs)))

    def __getitem__(self, index):
        frame_idx = index + 16
        frame_embs = [torch.tensor(self.vid_embs[x]) for x in list(range(frame_idx, frame_idx + self.frame_steps, self.frame_stride))]
        
        return torch.stack(frame_embs), frame_idx

    def __len__(self):
        return self.tot_frames
