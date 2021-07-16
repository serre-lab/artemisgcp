import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from models.baseline import StackedLSTM, MLP, StackedLSTMOne, BiStackedLSTMOne
from utils import bal_acc, class_report, plot_confusion_matrix, slackify
from torch.utils.data import DataLoader
from dataset_load_torch import MouseDataset
import pickle
import os
import time 
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

plt.ioff()
os.environ['CUDA_VISIBLE_DEVICES']='5'
train_file_path = '/media/data_cifs_lrs/projects/prj_nih/prj_andrew_holmes/Annot/'
val_file_path = '/media/data_cifs_lrs/projects/prj_nih/prj_andrew_holmes/Annot/'

traindata = MouseDataset(train_file_path) #object that allows us to get a data. __getitem__ returns 64 embs, labels
train_loader = DataLoader(traindata, batch_size=32, shuffle=True, pin_memory=True, num_workers=4, drop_last=True) #calls __getitem__ in batches of 32


valdata = MouseDataset(val_file_path,test=True)
val_loader = DataLoader(valdata, batch_size=32, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
baccs = []
test_loss = []
train_loss = []
all_labels_enc = []
loss_100 = []

if __name__ == '__main__':
    #creates biLSTM model. 
    model = BiStackedLSTMOne(input_size=1024, hidden_sizes=[256], num_classes=9, num_steps = 16)
    model = model.cuda()
    
    ##LOAD MODEL HERE
    model.load_state_dict(torch.load('/media/data_cifs_lrs/projects/prj_nih/prj_andrew_holmes/bootstrap_code/bi_model_bootstrapR6_bothvideos/model0.957272038308439.pth'))
    
    optimizer = optim.Adam(model.parameters(), lr= 1e-4) #1e-7

    class_weights = [162.3640201331385, 28.546959748786755, 16.66139055965611, 85.2006475249212, 13.16049220240837, 112.95606009262397, 14.006779281172088, 148.3239394838327, 36.45643456069996]
    class_weights = torch.FloatTensor(class_weights).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    #loss_fn = nn.CrossEntropyLoss()

    curr_iter = 0
    for epoch in range(5):
       for frames, labels in train_loader:
          model.train()
          curr_iter += 1
          print(frames.shape)
          time.sleep(4)
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
         
             """
             So close, So labels_t is a list [[32 preds randomly picked and -3 from 64 length]]
             When slackify checks y_true and y_ped are same, is looks in whol elist of 32. I need to 
             index individual items in list to check!
             """
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
                torch.save(model.state_dict(), 'bi_model_bootstrapR7_bothvideos/model'+ str(b_acc)+'.pth')
                print("model saved")
