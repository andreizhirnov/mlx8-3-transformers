import torch
import torch.nn as nn 
from torch.nn.functional import cosine_similarity, triplet_margin_loss
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 

## load data and split
images = np.load("data/images.npy")
labs = np.load("data/labs.npy")
full_dataset = TensorDataset(torch.tensor(images), torch.tensor(labs))
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])
 
testtt = torch.tensor(images[0:2])
# model

# model
class Attention(nn.Module):
    def __init__(self,  vk_i_dim, q_i_dim, v_dim, kq_dim):
        super().__init__()
        self.linear_v = nn.Linear(vk_i_dim, v_dim, bias=False)
        self.linear_k = nn.Sequential(
            nn.Linear(vk_i_dim, kq_dim, bias=False),
            nn.LayerNorm(kq_dim)
            )
        self.linear_q = nn.Sequential(
            nn.Linear(q_i_dim, kq_dim, bias=False),
            nn.LayerNorm(kq_dim)
            )
        self.k_scaler = torch.tensor(kq_dim**0.5)
        self.softmax_a = nn.Softmax(dim = -1)
        self.ln = nn.LayerNorm(vk_i_dim)
        self.dropout = nn.Dropout(0.1)               
        
    def forward(self, vals, keys, ques):
       v = self.linear_v(vals)
       k = self.linear_k(keys)
       q = self.linear_q(ques) 
       a = q @ k.transpose(-1,-2) / self.k_scaler
       return self.ln(self.dropout(self.softmax_a(a) @ v) + ques)

class Encodr(nn.Module):
    def __init__(self,  ext_dim, v_dim, kq_dim): 
       super().__init__()
       self.att = Attention(ext_dim, ext_dim, v_dim, kq_dim) 
       self.ff = nn.Sequential(
           nn.Linear(v_dim, ext_dim),
           nn.Dropout(0.1),
           nn.ReLU(),
           nn.Linear(ext_dim, ext_dim)
           ) 
       self.ln = nn.LayerNorm(ext_dim) 
 
    def forward(self, emb):
        x = self.att(emb, emb, emb)
        h = self.ff(x) + x
        return self.ln(h)
    
class Classifier(nn.Module):
    def __init__(self,  in_dim, seq_len, hidden_dim, v_dim, kq_dim, num_classes=10): 
       super().__init__()
       self.fc_start = nn.Linear(in_dim, hidden_dim)
       pos = torch.arange(0, seq_len).unsqueeze(0)
       self.register_buffer('pos', pos, persistent=False)  
       self.xtra_patch = nn.Embedding(1, hidden_dim)
       self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
       self.encoders = nn.ModuleList([Encodr(hidden_dim, v_dim, kq_dim) for i in range(2)])
       self.fc_final = nn.Linear(hidden_dim, num_classes)
   
    def forward(self, emb):
        x = self.fc_start(emb) + self.pos_embedding(self.pos)
        loc0 = torch.zeros((x.shape[0],1), dtype=torch.int64)
        x = torch.cat([self.xtra_patch(loc0), x], dim=1)
        for i, es in enumerate(self.encoders):
            x = self.encoders[i](x)
        x = self.fc_final(x[:,0,:])
        return x

def make(config):
    # Make the data 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size']) 
    
    # Make the model 
    model = Classifier(config['emb_dim'], config['seq_len'], config['hidden_dim'], config['v_dim'], config['kq_dim'])

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, config): 
    model.train()
    for epoch in range(config['num_epochs']):
        losses = 0
        for im, lab in train_loader:  
            im = im.to(config['device'])
            lab = lab.to(config['device'])
            
            with torch.autocast(config['device']):
                optimizer.zero_grad()
                p = model(im)
                loss = criterion(p, lab)
                loss.backward()
                optimizer.step()
                losses += loss
        mean_loss = losses/len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 
 

def test(model, test_loader, criterion):
    model.eval()
    losses = 0
    for im, lab in test_loader: 
        im = im.to(config['device'])
        lab = lab.to(config['device'])
        p = model(im)
        loss = criterion(p, lab)
        losses += loss
    mean_loss = losses/len(test_loader)
    print(f'Average test loss, Loss: {mean_loss.item():.4f}') 

config = dict(
    num_epochs=10,
    batch_size=256,
    learning_rate=0.01,
    emb_dim = images.shape[-1],
    seq_len = images.shape[1],
    hidden_dim = 32,
    v_dim = 32,
    kq_dim = 20,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

# execution
model, train_loader, test_loader, criterion, optimizer = make(config)  
print(model)

a = model(testtt)
print(f"test data shape: {a.shape}")

train(model, train_loader, criterion, optimizer, config)
test(model, test_loader, criterion) 

## model evaluation
test_ds_im = []
test_ds_la = []
for im, la in test_loader:
    test_ds_im.append(im)
    test_ds_la.append(la)
test_ds_im = torch.cat(test_ds_im)
test_ds_la = torch.cat(test_ds_la)
model.eval()
test_ds_pr = model(test_ds_im)
mca = MulticlassAccuracy(num_classes=10)
mca(test_ds_pr, test_ds_la )


















