import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.baseline import BiStackedLSTMOne
#from utils import bal_acc, class_report, plot_confusion_matrix, slackify
from dataset_load_inf import MouseDataset
import os
import csv

import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--embname", type=str, required=True)
parser.add_argument("--output_file_name", type=str, required=True)
#parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
test_loss = []
train_loss = []

BEHAVIOR_LABELS = {
    0: "drink",
    1: "eat",
    2: "groom",
    3: "hang",
    4: "sniff",
    5: "rear",
    6: "rest",
    7: "walk",
    8: "eathand"
}

bsize = 256
num_work_dataload = 8

def download_blob(bucket_name, source_blob_name, destination_folder):
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    print(destination_folder + os.sep + source_blob_name)
    blob.download_to_filename(destination_folder + os.sep + source_blob_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_folder + os.sep + source_blob_name
        )
    )

def parse_url(url: str):
    from urllib.parse import urlparse
    o = urlparse(url)
    return o.netloc, o.path.lstrip('/')


if __name__ == '__main__':
    #print('batch size-', bsize)
    #print('num workers dataloader-', num_work_dataload)
    
    model = BiStackedLSTMOne(input_size=1024, hidden_sizes=[256], num_classes=9, num_steps=16)
    model = model.cuda()
    model.load_state_dict(torch.load('/workspace/inference/models/model0.9573332767722087.pth'))
    model.eval()

    # emb_base_dir = '/cifs/data/tserre_lrs/projects/prj_nih/prj_andrew_holmes/inference/inference_i3d/'
    # emb_dir = emb_base_dir + exp_name
    # emb_file = args.embname

    # save_dir = '/cifs/data/tserre_lrs/projects/prj_nih/prj_andrew_holmes/inference/results_fallon_bsln/' + exp_name
    # if not os.path.exists(save_dir):
    #     try:
    #         os.mkdir(save_dir)
    #     except:
    #         pass

    #import pdb; pdb.set_trace()
    a = time.time()
    full_emb_file = args.embname
    save_file_name = args.output_file_name

    print(full_emb_file)

    # bucket_name, source_blob_name = parse_url(full_emb_file)
    # download_blob(bucket_name, source_blob_name, 'embeddings')
    # dirname = os.path.dirname(save_file_name)
    Path(dirname).mkdir(parents= True, exist_ok=True)
    # full_emb_file = 'embeddings' + os.sep + source_blob_name

    mousedata = MouseDataset(full_emb_file)
    inf_loader = DataLoader(mousedata, batch_size=bsize, shuffle=False, pin_memory=True, num_workers=num_work_dataload)
    print('loadpickle:', time.time() - a)
    #import pdb; pdb.set_trace()
    #print(full_emb_file, save_file_name)
    for frames, frame_start_list in inf_loader:
        #import pdb; pdb.set_trace()
        with torch.no_grad():

            #a = time.time()
            #frames_t, frame_start_list, vid_name = load_data(1000, frame_steps=64)
            #print('load data', time.time() - a)
            #print(frame_start_list[0], frame_start_list[-1], vid_name)

            #a = time.time()
            frames = frames.cuda(non_blocking=True)
            #torch.cuda.synchronize()
            #print('to gpu data', time.time() - a)

            #a = time.time()
            predictions = model(frames)
            #torch.cuda.synchronize()
            #print('forward', time.time() - a)

            _, preds = torch.max(F.softmax(predictions[:, :], dim=1), 1)  #:,:,-1
            #torch.cuda.synchronize()
            #a = time.time()
            with open(save_file_name, 'a+') as f:
                w = csv.writer(f)
                for key, val in zip(frame_start_list, preds):
                    w.writerow([key.item() + 64, val.item()])
        #print(frame_start_list.min(), frame_start_list.max())
    print('whole:', time.time() - a)
    # print("Test:  ", (b_acc), "max-bacc:", max(baccs), "ov acc:", accuracy_score(labels_t[:,-3].cpu().numpy(), preds.cpu().numpy()))
