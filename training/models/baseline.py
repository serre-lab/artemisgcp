import torch
from torch import nn
import torch.nn.functional as F

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, num_steps):
        super(StackedLSTM, self).__init__()
        self.num_steps = num_steps
        self.cfg = [input_size, *hidden_sizes]

        lstms = []
        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            lstm = nn.LSTMCell(input_size, hidden_size)
            lstms.append(lstm)

        self.lstms = nn.ModuleList(lstms)
        self.n_layers = len(lstms)
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        self.hiddens, self.cells = [], []

    def init_zero_state(self, batch_size, device):
        hiddens = []
        cells = []

        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            hiddens.append(torch.zeros(batch_size, hidden_size).to(device))
            cells.append(torch.zeros(batch_size, hidden_size).to(device))

        return hiddens, cells

    def forward(self, xs):
        batch_size = xs.size(0)
        device = xs[0].device

        self.hiddens, self.cells = self.init_zero_state(batch_size, device)
        #import pdb; pdb.set_trace()
        logits_t = []
        for t in range(self.num_steps):
            x = xs[:, -self.num_steps + t,:]
            
            for lstm in self.lstms:
                hx = self.hiddens[-self.n_layers]
                cx = self.cells[-self.n_layers]
                #import pdb; pdb.set_trace()
                h, c = lstm(x, (hx, cx))
                
                self.hiddens.append(h)
                self.cells.append(c)

                x = h
                #x = F.dropout(x, p = 0.7)
                logits_t.append(self.classifier(x))
        #x = F.dropout(x, p = 0.7)
        logits = self.classifier(x)
        
        return torch.stack(logits_t, dim=2)
    
class StackedLSTMOne(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, num_steps):
        super(StackedLSTMOne, self).__init__()
        self.num_steps = num_steps
        self.cfg = [input_size, *hidden_sizes]

        lstms = []
        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            lstm = nn.LSTMCell(input_size, hidden_size)
            lstms.append(lstm)

        self.lstms = nn.ModuleList(lstms)
        self.n_layers = len(lstms)
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)

    def init_zero_state(self, batch_size, device):
        hiddens = []
        cells = []

        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            hiddens.append(torch.zeros(batch_size, hidden_size).to(device))
            cells.append(torch.zeros(batch_size, hidden_size).to(device))

        return hiddens, cells

    def forward(self, xs):
        batch_size = xs.size(0)
        print('Batch size is: {}'.format(batch_size))
        device = xs[0].device
        print("xs is {}".format(xs))

        hiddens, cells = self.init_zero_state(batch_size, device)
        #import pdb; pdb.set_trace()
        
        for t in range(self.num_steps):
            x = xs[:, -self.num_steps - 9 + t, :]
            
            for lstm in self.lstms:
                hx = hiddens[-self.n_layers]
                cx = cells[-self.n_layers]
                #import pdb; pdb.set_trace()
                h, c = lstm(x, (hx, cx))
                
                hiddens.append(h)
                cells.append(c)

                x = h[:]
                
        x = F.dropout(x, p = 0.5)
        #import pdb; pdb.set_trace()
#        all_hidden = [hiddens[i] for i in range(5,99,3)]
#        mean_hidden = torch.mean(torch.stack(all_hidden, dim=2), dim=2)
        
        logits = self.classifier(x)
        
        return logits

class BiStackedLSTMOne(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, num_steps):
        super(BiStackedLSTMOne, self).__init__()
        self.num_steps = num_steps
        self.cfg = [input_size, *hidden_sizes]
        #creates a list with one layer of forward and backward lstm
        lstms = []
        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            lstm = nn.LSTMCell(input_size, hidden_size)
            lstms.append(lstm)
        
        lstms_rev = []
        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            lstm = nn.LSTMCell(input_size, hidden_size)
            lstms_rev.append(lstm)
        #makes thes models iterable
        self.lstms = nn.ModuleList(lstms)
        self.lstms_rev = nn.ModuleList(lstms_rev)
        self.n_layers = len(lstms)
        
        #self.classifier = nn.Linear(hidden_sizes[-1]*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(9)
        
        self.classifier1 = nn.Linear(256, 128)
        self.classifier2 = nn.Linear(256, 128)
	#only uses last classifier layer with input from for and rev lstm 
        self.classifier3 = nn.Linear(512, num_classes)

    def init_zero_state(self, batch_size, device):
        hiddens = []
        cells = []
        #initializes zero state for gradient descent
        for (input_size, hidden_size) in zip(self.cfg[:-1], self.cfg[1:]):
            hiddens.append(torch.zeros(batch_size, hidden_size).to(device))
            cells.append(torch.zeros(batch_size, hidden_size).to(device))

        return hiddens, cells

    def forward(self, xs):
        batch_size = xs.size(0)
        device = xs[0].device
        latents = []
        #how our data flows through (64 frames of video)
        hiddens, cells = self.init_zero_state(batch_size, device)
        #import pdb; pdb.set_trace()
        
        for t in range(self.num_steps):
            x = xs[:, 64-self.num_steps+1+t-3, :]
            #print(64-self.num_steps+1+t-3)
            for lstm in self.lstms:
                hx = hiddens[-self.n_layers]
                cx = cells[-self.n_layers]
                #import pdb; pdb.set_trace()
                h, c = lstm(x, (hx, cx))
                
                hiddens.append(h)
                cells.append(c)

                x = h[:]
        #x = F.dropout(x, p = 0.1)
        latents.append(x)
        
        hiddens, cells = self.init_zero_state(batch_size, device)
        for t in range(3):
            x = xs[:, -t-1, :]
            #print(64-t-1)
            for lstm in self.lstms_rev:
                hx = hiddens[-self.n_layers]
                cx = cells[-self.n_layers]
                #import pdb; pdb.set_trace()
                h, c = lstm(x, (hx, cx))
                
                hiddens.append(h)
                cells.append(c)

                x = h[:]
        #x = F.dropout(x, p = 0.1)
        latents.append(x)
        #import pdb; pdb.set_trace()
#        all_hidden = [hiddens[i] for i in range(5,99,3)]
#        mean_hidden = torch.mean(torch.stack(all_hidden, dim=2), dim=2)
        
        
#        out1 = F.relu(self.classifier1(self.bn1(latents[0])))
#        out2 = F.relu(self.classifier2(self.bn2(latents[1])))
        
        cat_latent = torch.cat(latents, dim=1)
        
        logits = self.classifier3(cat_latent)
#        logits = F.dropout(logits, p = 0.6)
#        logits = self.classifier3(logits)
        
        print("logits: {}".format(logits))       
        return logits

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        
        self.classifier1 = nn.Linear(input_size, 512)
        #self.classifier3 = nn.Linear(512, 256)
        self.classifier2 = nn.Linear(512, num_classes)



    def forward(self, xs):

        out = self.classifier1(xs[:,-1,:])
        #out = self.classifier3(out)
        
        logits = self.classifier2(out)
        
        return logits
