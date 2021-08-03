#--------------------------------------------------
# Dingdong Yang: 03-19-2019
#--------------------------------------------------
from __future__ import print_function

import glob
import os
import cv2
import random
import numpy as np
from pims import Video
import tensorflow as tf

class VideoIterator:
    def __init__(self, video_folder, sample_num=4, chunk_length=16, 
            batch_size=20, first_how_many=8000):
        """
            Input Variables:
            1. video_folder: full path of the root folder of videos
            2. sample_num: the sampled video number
            3. chunk_length: the length of the video chunk
        """

        self.video_folder = video_folder
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.first_how_many = first_how_many

        # Sampled video names 
        #all_videos = glob.glob(os.path.join(video_folder, "*.mp4"))
        #self.video_names = random.sample(all_videos, sample_num)
        self.video_names = [video_folder]
    def __call__(self):
        for video_path in self.video_names:
            # Initialize the video capture (handle)
            try:
                cap = Video(video_path)
            except:
                print("something wrong with video %s" % video_path)
                raise
                continue

            ret = True
            chunk_list = []
            batch_list = []
            frame_idx_list = []
            frame_idx_count = 0
            # Read and return numpy chunk
            while frame_idx_count < (108000-64): #len(cap): #cap.get_metadata()['nframes']:
                my_frame = cap.get_frame(frame_idx_count)
                #import pdb; pdb.set_trace()
                frame = my_frame[:,:,0]
                frame = cv2.resize(frame,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_LINEAR)
                #frame = cv2.copyMakeBorder(my_frame,(256-192)/2,(256-192)/2,0,0,cv2.BORDER_CONSTANT,None,[0,0,0])
                #frame = cv2.resize(my_frame,(256,256),interpolation = cv2.INTER_LINEAR)                
                #frame = frame[16:240,16:240]/127.5 - 1.
                frame = frame/127.5 - 1.
                #resized_frame = cv2.resize(my_frame,(256,256),interpolation = cv2.INTER_AREA)                
                #resized_frame = resized_frame[15:239,15:239]/127.5 - 1.
                chunk_list.append(frame)
                frame_idx_count += 1
                if len(chunk_list) < self.chunk_length: continue;
                else:
                    batch_list.append(np.asarray(chunk_list))
                    frame_idx_list.append(frame_idx_count)
                    chunk_list.pop(0)
                if len(batch_list) == self.batch_size:
                    yield (np.asarray(batch_list), os.path.basename(video_path), 
                            frame_idx_list)
                    batch_list = []
                    frame_idx_list = []
                if frame_idx_count - self.chunk_length > self.first_how_many:
                    break

            # Release the video hanlde at the end
            cap.close()

        # Yield dummy path at last in order to finish the last recording operation
        #yield (np.random.normal(size=[self.batch_size, self.chunk_length, 480, 640, 3]),
        #        "dummy_name", [0, ] * self.batch_size)
        yield (np.random.normal(size=[self.batch_size, self.chunk_length, 192, 256, ]),
                "dummy_name", [0, ] * self.batch_size)

# By RohitSaha
def preprocess_for_eval(volume,
                    num_frames,
                    height,
                    width,
                    target_image_size,
                    list_of_augmentations=[]):

    
    ##### Crop center 224*224 patch
    #height, width = get_shape[1], get_shape[2]
    center_x = tf.cast(
        tf.divide(
            height,
                #if type(height) == int else height[0],
            2),
        tf.int32)

    center_y = tf.cast(
        tf.divide(
            width,
                #if type(width) == int else width[0],
            2),
        tf.int32)

    offset_height = tf.subtract(
        center_x,
        112)
    offset_width = tf.subtract(
        center_y,
        112)
    target_height, target_width = target_image_size,\
        target_image_size
    volume = tf.image.crop_to_bounding_box(
        volume,
        offset_height,
        offset_width,
        target_height,
        target_width)
    
    return volume

def preprocessing_raw(video_chunk):
    """
        Input Variable:
        1. video_chunk: the tf tensor of chunks, 4D tensor, channel last
    """
    shape = video_chunk.get_shape().as_list()
    raw_length, raw_width = shape[1], shape[2]

    # Zero padding the shorter side of videos
    pad_size = int(abs((raw_length - raw_width) / 2.0))
    if raw_length <= raw_width:
        paddings = tf.constant([[0, 0], [pad_size, pad_size], [0, 0], 
            [0, 0]])
    elif raw_length > raw_width:
        paddings = tf.constant([[0, 0], [0, 0], [pad_size, pad_size], 
            [0, 0]])

    video_chunk = tf.pad(video_chunk, paddings, "CONSTANT", 
            constant_values=0)

    # Resize
    video_chunk = tf.cast(video_chunk, tf.float32) / 127.5 - 1.0 # [-1.0, 1.0]
    video_chunk = tf.image.resize_bilinear(video_chunk, size=[256, 256])
    video_chunk = preprocess_for_eval(video_chunk, 16, 256, 256, 
            target_image_size=224)

    # Expand the dim
    # video_chunk = tf.expand_dims(video_chunk, 0)

    return video_chunk

def preprocessing_raw_new(video_chunk):

    video_chunk = tf.expand_dims(video_chunk, -1)
    video_chunk = tf.broadcast_to(video_chunk, [16, 192, 256, 3])

    return video_chunk
# Smooth out the single frame prediction
def postprocessing_predicted_labels(preds_behaviors, frame_idx_list):
    """
        Functions: (in order)
            1. Smooth out single frames
            1. Format the labeling to (start, end, class)
            2. Consecutive rest frames < 150 will inherit previous class
            2. Smooth out 2 frame label prediction (eathand)
    """

    csv_list = []
    pre_label = None
    start_i = 0
    end_i = 0
    for i, idx in enumerate(frame_idx_list):
        if i > 0 and preds_behaviors[i] != pre_label:
            if i < len(frame_idx_list) - 1 and \
                    preds_behaviors[i] == preds_behaviors[i + 1]:
                end_i = i - 1
                csv_list.append([frame_idx_list[start_i], frame_idx_list[end_i],
                    pre_label])
                start_i = i
                pre_label = preds_behaviors[i]
            elif i == len(frame_idx_list) - 1:
                break

        elif i == 0 :
            pre_label = preds_behaviors[0]

    # Smooth out consecutive frames < 150, and only few (< 3 frames) result
    pre_label = None
    pre_elements = []
    new_list = []
    for csv_elements in csv_list:
        if csv_elements[2] == "rest" and pre_label is not None and \
                (csv_elements[1] - csv_elements[0]) <= 150:
            # If this is short rest then expand the elements
            pre_elements[1] = csv_elements[1]
        elif pre_label is not None and \
             (csv_elements[1] - csv_elements[0]) <= 3:
            # Chunk length equal or less than 3, then merge with previous one
            pre_elements[1] = csv_elements[1]
        elif pre_label is None:
            pre_label = csv_elements[2]
            pre_elements = csv_elements
        else:
            new_list.append(pre_elements)
            pre_label = csv_elements[2]
            pre_elements = csv_elements

    csv_lines = []
    for csv_elem in new_list:
        csv_lines.append("%d,%d,%s\n" % (csv_elem[0], csv_elem[1], csv_elem[2]))

    return csv_lines

def _mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
