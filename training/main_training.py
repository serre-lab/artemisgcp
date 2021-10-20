
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from sklearn.metrics import accuracy_score
from baseline import StackedLSTM, MLP, StackedLSTMOne, BiStackedLSTMOne
from training.upload_blob import download_blob
from utils import bal_acc, class_report, plot_confusion_matrix, slackify
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from download_blobs import downloadData
from upload_blob import upload_blob
from dataset_load_torch import MouseDataset
import pickle
import os
import time
import argparse
import logging
from urllib.parse import urlparse
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logging.info('The video main training code is running')


plt.ioff()


def download_yaml(bucket_name, source_blob_name) -> str:

   client = storage.Client()

   model_bucket = client.bucket(bucket_name)
   yaml_blob = model_bucket.blob(source_blob_name)

   document = yaml_blob.download_as_text()

   return document

def update_and_upload_yaml(raw_txt, version, accuracy, model_path):

   import yaml

   document = yaml.safe_load(raw_txt)

   current_model = {'path': model_path, 'accuracy': accuracy}
   document[version] = current_model

   with open('trained_models/updated_models.yaml', 'w') as f:
      dump = yaml.dumps(document)
      f.wrte(dump)

   upload_blob(args.save, 'trained_models/updated_models.yaml', 'trained_models/models.yaml')


#Get video path argument
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Path to model URI folder.')
parser.add_argument('-e', '--emb', help='Path to embs URI folder.', required=True)
parser.add_argument('-a', '--annotation', help='Path to annotation URI foler', required=True)
parser.add_argument('-s', '--save', help='Path to annotation URI foler', required=True)

args = parser.parse_args()
parsedSave= urlparse(args.save)

# if os.path.isfile(args.model) == False:
#    model_accuracy = None
#    model_exists = False
#    for file in glob.glob('models/*.pth'):
#       res = re.findall("\d+\.\d+", file)
#       model_exists = True

#       if model_accuracy == None:
#             model_accuracy= float(res[0])
#             model_name = file
      
#       else:
#             if model_accuracy > float(res[0]):
#                 continue
#             else:
#                 model_accuracy = float(res[0])
#                 model_name = file
#    model_path = file
# else:
#    model_path = args.model
   

#download blobs to container based on argument
downloadData(annotation_bucket_name=args.annotation, embedding_bucket_name=args.emb)

#f = open('annotations/Trap2_FC-A-1-12-Postfear_new_video_2019Y_02M_23D_05h_30m_06s_cam_6394846-0000.mp4_training_annotations.json')

#print(f.read())

#load available data frames
data = MouseDataset('annotations/', 'embeddings/') #object that allows us to get a data. __getitem__ returns 64 embs, labels

#split training and validaion set
validation_split = .2
shuffle_dataset = True
random_seed= 42


# Creating data indices for training and validation splits:
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(data, batch_size=32, pin_memory=True, num_workers=4, drop_last=True, sampler=train_sampler) #calls __getitem__ in batches of 32
val_loader = DataLoader(data, batch_size=32, pin_memory=True, num_workers=4, drop_last=True, sampler=valid_sampler) #calls __getitem__ in batches of 32


#set up arrays for training
baccs = []
test_loss = []
train_loss = []
all_labels_enc = []
loss_100 = []

if __name__ == '__main__':
    #creates biLSTM model. 
    model = BiStackedLSTMOne(input_size=1024, hidden_sizes=[256], num_classes=9, num_steps = 16)
    model = model.cuda()

    model_path = args.model
    
    ##LOAD MODEL HERE
    model.load_state_dict(torch.load(model_path))
    
    optimizer = optim.Adam(model.parameters(), lr= 1e-4) #1e-7

    class_weights = [162.3640201331385, 28.546959748786755, 16.66139055965611, 85.2006475249212, 13.16049220240837, 112.95606009262397, 14.006779281172088, 148.3239394838327, 36.45643456069996]
    class_weights = torch.FloatTensor(class_weights).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    curr_iter = 0
    for epoch in range(200):
       for frames, labels in train_loader:
          model.train()
          curr_iter += 1
          
          frames = frames.cuda()
          labels = labels.cuda()
          predictions = model(frames)
          loss = loss_fn(predictions, labels[:,-3])
	        
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          loss_100.append(loss.cpu().detach().numpy())

          print('Iter: %d, loss: %f'%(curr_iter, loss_100[-1]))

	  #VALIDATION ------
          if (curr_iter) % 100 == 0:
             _, preds = torch.max(F.softmax(predictions[:,:], dim=1),1)
             #print("Iter: ", curr_iter, "Training:  ", np.average(loss_100), bal_acc(labels[:,-3].cpu().numpy(), preds.cpu().numpy()))
	            
             model.eval()
             all_val_preds = []
             all_labels = []
             with torch.no_grad():
                for frames_t, labels_t in val_loader:
                   frames_t = frames_t.cuda()
                   labels_t = labels_t.cuda()
                   optimizer.zero_grad()
                   predictions = model(frames_t)
                   loss = loss_fn(predictions, labels_t[:,-3])
                   
		   
                   _, preds = torch.max(F.softmax(predictions[:,:], dim=1),1)
                   all_labels.append(labels_t[:,-3].cpu().numpy())
                   all_val_preds.append(preds.cpu().numpy())             
         
    
             all_labels = np.concatenate(all_labels)
             all_val_preds = np.concatenate(all_val_preds)
             all_val_preds_slack = slackify(all_labels, all_val_preds)
             all_val_preds_slack = torch.tensor(all_val_preds_slack)
            
             
             
             b_acc = bal_acc(all_labels, all_val_preds_slack.cpu().numpy()) #STILL FINDING LAST ONE THink it might have to do with how I am loading (data_load_torch.py)
             baccs.append(b_acc)
             
             print("Test:  ", loss.cpu().detach().numpy(), (b_acc), "max-bacc:", max(baccs), "ov acc:", accuracy_score(all_labels, all_val_preds_slack.cpu().numpy()))
             #class_report(labels_t[:,-1].cpu().numpy(), preds.cpu().numpy())
             test_loss.append(loss.cpu().detach().numpy())
             train_loss.append(np.average(loss_100))
	            
	           # fig = plt.figure()
	           # plt.plot(test_loss, 'b')
	           # plt.plot(train_loss, 'r')
	           # #plt.plot(baccs, 'm')
	           # fig.savefig('plots.png', dpi=300.)
	           # plt.close()
	           # print("--------------------------------------------")

             loss_100 = []
             model.train()
             if b_acc==max(baccs) and b_acc>0.7:
               #  model_name = 'model_acc_{}.pth'.format(b_acc)
               #  model_path = 'trained_models/' + model_name
               #  torch.save(model.state_dict(), model_name)
               #  upload_blob(args.save, model_name, model_path)
               #  print("model saved")
               dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
               version = 'LSTM_model_{}.pth'.format(dt)
               model_path = 'trained_models/' + version
               document = download_yaml(args.annotations, source_blob_name = 'trained_models/models.yaml')
               update_and_upload_yaml(document, version, b_acc, model_path)
               torch.save(model.state_dict(), version)
               upload_blob(args.save, version, model_path)
               print("model saved")
               


     

   
