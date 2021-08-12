#--------------------------------------------------
# Dingdong Yang: 03-19-2019
#--------------------------------------------------
from __future__ import print_function

import glob
import os
#import mkl
#mkl.set_num_threads(1)
import numpy as np
import sys
#sys.path.append('/media/data_cifs/didoyang/self_tpu_projects/')
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from absl import flags
# from absl import app
import time
from central_reservoir.models import i3d
from read_videos import VideoIterator, preprocessing_raw, preprocessing_raw_new,\
        postprocessing_predicted_labels, _mkdir
# from tqdm import tqdm
# import dask.dataframe as dd

# FLAGS = flags.FLAGS

# flags.DEFINE_string('model_folder_name', default="/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/i3d_full_processed_nih/",
#     help='To mention the model path')

# flags.DEFINE_string('video_name', default=None,
#     help='The video folder format')

# flags.DEFINE_integer('step', default=24600,
#     help='To specify the checkpoint')
# flags.DEFINE_integer('how_many_per_folder', default=1,
#     help='How many videos to process per folder')
# flags.DEFINE_integer('batch_size', default=50,
#     help='Processing batch size')
# flags.DEFINE_integer('first_how_many', default=8000,
#     help='Process first how many frames of each video')

# flags.DEFINE_string('base_result_dir', default="./result_dir", 
#     help='Base result directory')
# flags.DEFINE_string('exp_name', default="..", 
#     help='current experiment name')

BEHAVIOR_INDICES = {
    0:"drink",
    1:"eat",
    2:"groom",
    3:"hang",
    4:"sniff",
    5:"rear",
    6:"rest",
    7:"walk",
    8:"eathand"}

def run_i3d(model_folder_name = "models", 
            step = 24600,
            video_name = None,
            how_many_per_folder = 1,
            batch_size = 50,
            first_how_many = 8000,
            base_result_dir= None,
            exp_name = None):

    global_time = time.time()
    ckpt_path = os.path.join(
        model_folder_name,
        'model.ckpt-{}'.format(step))
    meta_path = os.path.join(
        model_folder_name,
        'model.ckpt-{}.meta'.format(step))

    video_folders = glob.glob(video_name)
    _mkdir(os.path.join(base_result_dir, exp_name))

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                        allow_soft_placement=True, device_count = {'CPU': 1})
    session = tf.Session(config=config)

    with session.as_default() as sess:
        # Hard coded input array
        '''
        pre_input = tf.placeholder(tf.uint8, 
                shape=[FLAGS.batch_size, 16, 480, 640, 3])
        input_chunk = tf.map_fn(preprocessing_raw,
                pre_input, dtype=tf.float32, back_prop=False)
        '''
        #### debug to see cost of preprocess input
        #input_chunk = tf.constant(0.,shape=[FLAGS.batch_size,16,224,224,3],dtype=tf.float32)

        pre_input = tf.placeholder(tf.float32,
                shape=[batch_size, 16, 192, 256,])
        input_chunk = tf.map_fn(preprocessing_raw_new,
                pre_input, dtype=tf.float32, back_prop=False)
        network = i3d.InceptionI3d(
            final_endpoint='Logits',
            use_batch_norm=True,
            use_cross_replica_batch_norm=False,
            num_classes=9,
            spatial_squeeze=True,
            dropout_keep_prob=1.0)

        logits, end_points = network(
            inputs=input_chunk, #input_chunk, 
            is_training=False)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        # sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        all_preds = {}
        frame_idx_list = []
        pre_name = None
        for video_folder in video_folders:
            start_outer_loop = time.time()
            video_iterator = VideoIterator(video_folder, 
                    sample_num=how_many_per_folder, chunk_length=16, 
                    batch_size=batch_size, 
                    first_how_many=first_how_many)()
            print ('Time elapsed init: %f'%(time.time() - start_outer_loop))
            chunk_count = 0
            time_read = []
            time_batch = []
            #pbar = tqdm(total=108000)
            while True:
                try: 
                    start = time.time()
                    chunk_array, video_name, frame_idx = next(video_iterator)
                    
                    time_read.append(time.time() - start)
                    #print ('Time elapsed batch read: %f %f'%(time_read[-1], np.mean(time_read)))
                    # If reach the next video, write the result
                    if video_name != pre_name and pre_name is not None:
                        #preds_behav = [BEHAVIOR_INDICES[i] for i in all_preds]

                        # Record the full video path for uploading the videos
                        #with open(os.path.join(FLAGS.result_dir, "video_selected.txt"), "a") as f:
                        #    f.write(os.path.join(video_folder, pre_name + "\n"))

                        # Record the csv files
                        # The way of naming the csv files is related to 
                        # the name matching algorithm in BABAS
                        vid_path_elem = video_folder.split('/')
                        pickle_file = os.path.join(base_result_dir, exp_name, pre_name.rstrip(".mp4") + ".p")
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(all_preds, f)

                        # Refresh the loop dependent variable
                        pre_name = video_name
                        all_preds = {}
                        frame_idx_list = []
                        chunk_count = 0
                    elif pre_name is None:
                        pre_name = video_name
                    #import pdb
                    #pdb.set_trace()
                    preds = sess.run(end_points['Avg_pool'], feed_dict={pre_input: chunk_array})

                    #preds_max = list(np.argmax(preds, axis=-1))
                    #all_preds += preds_max
                    
                    for i,frame_no in enumerate(frame_idx):
                        #all_preds[frame_no] = [preds[0][i].squeeze(), preds[1][i]]
                        all_preds[frame_no] = preds[i].squeeze()
                    frame_idx_list += frame_idx
                    chunk_count += batch_size
                    
                    time_batch.append(time.time() - start)
                    print("vid: %s with %d chunks READ: \033[1;33m %f (%f)\033[0;0m WHOLE: \033[1;33m %f (%f)\033[0;0m" % (os.path.join(exp_name, pre_name), chunk_count, time_read[-1], np.mean(time_read), time_batch[-1], np.mean(time_batch)))
                    #pbar.update(FLAGS.batch_size)
                    
                    #print ('Time elapsed one batch: %f %f'%(time_batch[-1], np.mean(time_batch)))
                    #sys.stdout.flush()
            
                except StopIteration:
                    break
    #pbar.close()
    print('Finished whole thing in: ', time.time()-global_time)

    return pickle_file

# def main(unused_argv):

#     JSON_file = FLAGS.video_name

#     dataframe = dd.read_json(JSON_file).compute()

#     for index, row in dataframe.iterrows():
#         run_i3d(model_folder_name = FLAGS.model_folder_name, 
#                 video_name = row['videoGcsUri'],
#                 batch_size = FLAGS.batch_size,
#                 first_how_many = FLAGS.first_how_many,
#                 base_result_dir = FLAGS.base_result_dir,
#                 exp_name = FLAGS.exp_name)

# if __name__ == '__main__':
    
#     app.run(main)
    
