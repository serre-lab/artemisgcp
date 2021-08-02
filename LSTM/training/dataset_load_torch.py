#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:49:24 2019

@author: alekhka & jbeck
"""
import pickle
#import os
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import glob
import numpy as np

##create tuple with video index and available frame

class MouseDataset(Dataset):
  def __init__(self, data_path, test=False):
    
    """
    [returns]:
      - dict with sructure {i:{frame:label}} *note i is index that pairs label and emb title
      - dict with structure {i:{frame:[1024 embs]}}
      - dict with structure {"accumilated index:[i,acceptable frame]} *note "accumilated index" need to be different for __getitem__
    """
    self.BEHAVIOR_LABELS = {
            "drink": 0,
            "groom": 2,
            "eat": 1,
            "hang": 3,
            "sniff": 4,
            "rear": 5,
            "rest": 6,
            "walk": 7,
            "eathand": 8,
            
        }
    #set number of frames associated with label (temporal)
    self.frame_steps = 64
    self.frame_stride = 1
    #empty dictionary to fill with available frames from training dataset
    self.avail_frames = {}
    self.label_dict = {}
    self.emb_dict = {}
    #data path to ANNOT (where artemis files are saved). Argument passed to class to determine test or train folder
    if test == False:
      self.label_path = data_path + "pickle_files/train/"
    else:
      self.label_path = data_path + "pickle_files/test/"
    self.emb_path = data_path + "embs/"
    #set to 0 to use to randomly pick key and keep key conistent between videos and embs
    i = 0 
    #accumilator for pytorch indexer
    ind_for_dict_frames = 0
    #load all label files into one dataframe
    for file in glob.glob(self.label_path + '*'):#[0:1]:     
      #load labels and embs
      print('attempting to load file:', file)
      if test == False:
        labels = pd.read_pickle(file)
        print("the length of the train labels file is: "+ str(len(labels)) + " : Now loading file")
        #try:
        embs = pd.read_pickle(self.emb_path + file[file.rfind("/")+1:-8] + ".p")
        #except:
            #continue
      else:
        labels = pd.read_pickle(file)
        print("the length of the test labels file is: "+ str(len(labels)) + " : Now loading file")
        #try:
        embs = pd.read_pickle(self.emb_path + file[file.rfind("/")+1:-7] + ".p")
        #except:
            #continue
      #print(file)
      labels = labels[labels.pred != 'none']
      #iterate through label file and find all available files. These set start for first and last frame
      print(labels.size)
      if labels.size == 0:
        continue 
      frame_one = max(16, labels['frame'].iloc[0])
      print(frame_one)
      frame_lst = max(16, labels['frame'].iloc[0])
      for index, row in labels.iterrows():     
        if row['pred'] != 'none':
	  #if diff between next frame and last frame index greater than 1, this signals skip
          if row['frame'] - frame_lst > 1:
	    #make sure block of cont frames greater than 64
            if frame_lst - frame_one >= 64:
              for frame in np.arange(frame_one,frame_lst-63): #one less because np.arange doesn't include last number
                self.avail_frames[ind_for_dict_frames] = [i,frame]
                ind_for_dict_frames += 1
                #set new frame 1 and updater for new block of cont frames
                frame_one = row['frame']
                frame_lst = row['frame']
            else:
	      #if skip in frames, but they ar enot 64 frames long, still continue
              frame_one = row['frame']
              frame_lst = row['frame']
          else:
            frame_lst = row['frame']

      #if you get to end of file and no skips from new block to end, add frames if lenth greater than 64
      if frame_lst - frame_one >= 64:
        for frame in np.arange(frame_one,frame_lst-(self.frame_steps-1)): #one less because np.arange doesn't include last number
          self.avail_frames[ind_for_dict_frames] = [i,frame]
          ind_for_dict_frames += 1
          #append file title to video_files and labels to video_labels 
      
      self.label_dict.update({i:dict(zip(labels.frame,labels.pred))}) #dict with sructure {i:{frame:label}}
      #append embs to file organized like labels. Don't make another file name dict - used as fail safe
      #organized by frame:1024 array of embs
      self.emb_dict.update({i:embs}) #dict with structure {i:{frame:[1024 embs]}}
     
      self.tot_frames = len(self.avail_frames)
      i += 1
      print("The total available frames are: " + str(self.tot_frames))

  def __getitem__(self, index):
    #randomly choose index inside accumilator in avail frames dict
    avail_frame_idx = index
    #get list with [video index, and available frame]
    video_frame_idx = self.avail_frames[avail_frame_idx]
    #set frame to pick out correct label. Default is 64 up
    video_idx_plus = video_frame_idx[1] + self.frame_steps
    #get frame embs and labels for avail frame and 64 more
    frame_embs = [torch.tensor(self.emb_dict[video_frame_idx[0]][x]) for x in list(range(video_frame_idx[1], video_idx_plus, self.frame_stride))]
    frame_labels = [torch.tensor(self.BEHAVIOR_LABELS[self.label_dict[video_frame_idx[0]][x]]) for x in list(range(video_frame_idx[1], video_idx_plus, self.frame_stride))]
    return torch.stack(frame_embs), torch.stack(frame_labels)
   

  def __len__(self):
    return self.tot_frames
  
  def get_all():
    print('loaded all frames: ' + str(selt.tot_frames))
    
    
#NEEr TO GET IT TO SKIP NONE. gETTING KEY ERROR IN TRAINING
