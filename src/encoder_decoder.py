import os
os.chdir("C:/Users/andre/Dropbox/MLX/week 3/gitrepo/mlx8-3-transformers")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from src.data_prep_functions import break_and_flatten_3D

torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 
 

## load data and split
images = np.load("data/comb_images.npy")[:700]
labs = np.load("data/comb_labs.npy")[:700]
vocab = np.load("data/vocab.npy")
lookup = {w: i for i, w in enumerate(vocab)}

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
        self.register_buffer("k_scaler", torch.tensor(kq_dim**0.5))
        self.softmax_a = nn.Softmax(dim = -1)
        self.ln = nn.LayerNorm(q_i_dim)
        self.dropout = nn.Dropout(0.1)               
        
    def forward(self, vals, keys, ques, key_mask = None):
       v = self.linear_v(vals)
       k = self.linear_k(keys)
       q = self.linear_q(ques) 
       a = q @ k.transpose(-1,-2) / self.k_scaler
       if key_mask is not None:
           a = a.masked_fill(key_mask.unsqueeze(1), float('-inf')) 
       return self.ln(self.dropout(self.softmax_a(a) @ v) + ques)
            
class Encodr(nn.Module):
    def __init__(self,  ext_dim, v_dim, kq_dim): 
       super().__init__()
       self.att = Attention(ext_dim, ext_dim, v_dim, kq_dim) 
       self.ff = nn.Sequential(
           nn.Linear(v_dim, ext_dim),
           nn.ReLU(),
           nn.Linear(ext_dim, ext_dim)
           ) 
       self.ln = nn.LayerNorm(ext_dim) 
 
    def forward(self, emb):
        x = self.att(emb, emb, emb)
        h = self.ff(x) + x
        return self.ln(h)

class Decodr(nn.Module):
    def __init__(self,  enc_dim, dec_dim, v_dim, kq_dim): 
       super().__init__()
       self.selfatt = Attention(dec_dim, dec_dim, v_dim, kq_dim)
       self.att = Attention(enc_dim, dec_dim, v_dim, kq_dim)       
       self.ff = nn.Sequential(
           nn.Linear(v_dim, dec_dim),
           nn.ReLU(),
           nn.Linear(dec_dim, dec_dim)
           ) 
       self.ln = nn.LayerNorm(dec_dim)
       
    def forward(self, enc, dec, mask): 
        d = self.selfatt(dec, dec, dec, key_mask = mask) 
        d = self.att(enc, enc, d)
        d = self.ln(self.ff(d) + d)
        return d

# in_dim is the dimension of a patch
# seq_len is the number of patches

class Annotator(nn.Module):
    def __init__(self,  in_dim, seq_len, rseq_len, vocab_len, hidden_dim, v_dim, kq_dim): 
       super().__init__()
       self.fc_start = nn.Linear(in_dim, hidden_dim)
       pos = torch.arange(0, seq_len).unsqueeze(0)
       self.register_buffer('pos', pos, persistent=False)
       self.xtra_patch = nn.Embedding(1, hidden_dim)
       self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
       self.encoders = nn.ModuleList([Encodr(hidden_dim, v_dim, kq_dim) for i in range(2)])
       self.r_embedding = nn.Embedding(vocab_len, hidden_dim, padding_idx=lookup['<p>'])
       rpos = torch.arange(0, rseq_len).unsqueeze(0)
       self.register_buffer('rpos', rpos, persistent=False)  
       self.rpos_embedding = nn.Embedding(rseq_len, hidden_dim)
       self.decoders = nn.ModuleList([Decodr(hidden_dim, hidden_dim, v_dim, kq_dim) for i in range(2)])
       self.fc_final = nn.Linear(hidden_dim, vocab_len-2)
       
    def forward(self, fig_emb, prequel, mask):
## run the encoder
        x = self.fc_start(fig_emb) + self.pos_embedding(self.pos)
        loc0 = torch.zeros((x.shape[0],1), dtype=torch.int64)
        x = torch.cat([self.xtra_patch(loc0), x], dim=1)
        for i, es in enumerate(self.encoders):
            x = self.encoders[i](x)
## run the decoder
        d = self.r_embedding(prequel) + self.pos_embedding(self.rpos)
        for i, de in enumerate(self.decoders):
            d = de(x, d, mask = mask) 
        pick_cols = torch.sum(mask == False, dim=1)
        pick_rows = torch.arange(len(mask))
        return self.fc_final(d[pick_rows,pick_cols,:])

def make(config):
    # Make the data
    seq_len = labs.shape[1]
    sel_preq = np.tril(np.ones(seq_len, dtype=np.int64), k=-1)==0 
    counts = np.sum(np.logical_and(labs !=lookup['<p>'], labs !=lookup['<s>']), 1)
    nimages = np.repeat(break_and_flatten_3D(images, config['patch_dim']), counts, axis=0)
    infld = np.stack((labs,) * seq_len, axis=1) 
    infld[:,sel_preq] = lookup['<p>']
    infld = np.concatenate([np.expand_dims(labs, -1), infld], -1)
    infld = infld.reshape(infld.shape[0]*seq_len, seq_len+1)
    nlabs = infld[np.logical_and(infld[:,0] !=lookup['<p>'], infld[:,0] !=lookup['<s>']),:]
    
    full_dataset = TensorDataset(torch.tensor(nimages), 
                                 torch.tensor(nlabs[:,1:], dtype = torch.long), 
                                 torch.tensor(nlabs[:,0], dtype = torch.long))
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], 
                                               shuffle=True,
                                               pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                              pin_memory = True)
    
    # Make the model 
    model = Annotator(in_dim = config['patch_dim']**2,
                      seq_len = config['seq_len'],
                      rseq_len = config['rseq_len'],
                      vocab_len = len(config['vocab']),
                      hidden_dim = config['hidden_dim'],
                      v_dim = config['v_dim'],
                      kq_dim = config['kq_dim'])
    
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, config): 
    model = model.to(config['device'], non_blocking=True)
    model.train()
    for epoch in range(config['num_epochs']):
        losses = 0
        for im, preq, lab in train_loader:
            im = im.to(config['device'], non_blocking=True)
            preq = preq.to(config['device'], non_blocking=True)
            lab = lab.to(config['device'], non_blocking=True)
            mask = (preq == lookup["<p>"]).to(config['device'], non_blocking=True)
            with torch.autocast(config['device']):
                optimizer.zero_grad()
                p = model(im, preq, mask)
                loss = criterion(p, lab)
                loss.backward()
                optimizer.step()
                losses += loss
        mean_loss = losses/len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 
 
def test(model, test_loader, criterion):
    model.eval()
    mca = MulticlassAccuracy(num_classes=11)
    losses = 0
    for im, preq, lab in test_loader: 
        im = im.to(config['device'], non_blocking=True)
        preq = preq.to(config['device'], non_blocking=True)
        lab = lab.to(config['device'], non_blocking=True)
        mask = (preq == lookup["<p>"]).to(config['device'], non_blocking=True)
        p = model(im, preq, mask)
        loss = criterion(p, lab)
        losses += loss
        mca.update(p, lab)
    mean_loss = losses/len(test_loader)
    print(f'Average test loss, Loss: {mean_loss.item():.4f}') 
    print("Accuracy:", mca.compute().item()) 

config = dict(
    num_epochs=10,
    batch_size=256,
    learning_rate=0.01,
    patch_dim = 4, 
    rseq_len = labs.shape[1],
    vocab = vocab,
    hidden_dim = 32,
    v_dim = 32,
    kq_dim = 20, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )
config['seq_len'] = int(np.ceil(images[0].size/config['patch_dim']**2))

# execution
model, train_loader, test_loader, criterion, optimizer = make(config)  
print(model)

def pick(n):
    flabs = labs[:n]
    fimages = images[:n]
    sel_preq = np.tril(np.ones(flabs.shape[1], dtype=np.int64), k=-1)==0 
    counts = np.sum(np.logical_and(flabs!=lookup['<p>'], flabs !=lookup['<s>']), 1)
    nimages = np.repeat(break_and_flatten_3D(fimages, config['patch_dim']), counts, axis=0)
    infld = np.stack((flabs,) * flabs.shape[1], axis=-1) 
    infld[:,sel_preq] = lookup['<p>']
    infld = np.concatenate([np.expand_dims(flabs, -1), infld], -1)
    infld = infld.reshape(flabs.shape[0]*flabs.shape[1], flabs.shape[1]+1)
    nlabs = infld[np.logical_and(infld[:,0] !=lookup['<p>'], infld[:,0] !=lookup['<s>']),:]
    return torch.tensor(nimages), torch.tensor(nlabs[:,1:]), torch.tensor(nlabs[:,0])

test_v, test_s, test_t = pick(2) 
mask = (test_s == lookup["<p>"])

model = Annotator(in_dim = config['patch_dim']**2,
                  seq_len = config['seq_len'],
                  rseq_len = config['rseq_len'],
                  vocab_len = len(config['vocab']),
                  hidden_dim = config['hidden_dim'],
                  v_dim = config['v_dim'],
                  kq_dim = config['kq_dim'])
    
a = model(test_v, test_s, mask)
print(f"test data shape: {a.shape}")

train(model, train_loader, criterion, optimizer, config)
torch.save(model.state_dict(), "temp/current_model.pth")

test(model, test_loader, criterion) 
 


















