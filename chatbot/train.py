#!/usr/bin/env python
# coding: utf-8

# In[35]:


import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data  import Dataset
from torch.utils.data  import DataLoader
from NLP_utils import tokenize, stem, vectorize


# In[36]:


with open('intents.json','r') as f:
    intents = json.load(f)


# In[37]:


total_words = []
tags = []
patterns = []
xy = []
for intent in intents['intents']:
    tags.append(intent['tag']) 
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        total_words.extend(w)
        xy.append((w,intent['tag']))
ignore = ['!','.',';','/','@','?']
total_words = sorted(set([stem(w) for w in total_words if w not in ignore]))
tags = sorted(tags)


# In[38]:


x_tr, y_tr = [] , []
for (x,y) in  xy:
    x_vec = vectorize(x,total_words)
    print(x_vec)
    x_tr.append(x_vec)
    y_tr.append(tags.index(y))
print(x_tr, y_tr)


# In[39]:


x_tr, y_tr = np.asarray(x_tr), np.asarray(y_tr)
class BotDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_tr)
        self.x_data = x_tr
        self.y_data = y_tr

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# In[40]:


dataset = BotDataset()
train_dl = DataLoader(dataset=dataset, batch_size=8, shuffle=True)


# In[41]:


from model import BotNet


# In[42]:


input_size = x_tr.shape[1]
hidden_size = 8
output_size = len(tags)
print(input_size, hidden_size, output_size)
assert len(total_words) == input_size, 'input vector and input size does not match.'


# In[ ]:





# In[65]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BotNet(input_size, hidden_size, output_size).to(device)


# In[68]:


loss = nn.functional.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000
batch_size = 8


# In[69]:


for epoch in range(epochs):
    for (x,y) in train_dl:
        x = x.to(device).float()
        y = y.to(dtype=torch.long).to(device)
        y_pred = model(x)
        l = loss(y_pred,y)
        opt.zero_grad()
        l.backward()
        opt.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch : {epoch+1}/{epochs}, Loss = {l.item():.4f}')
        
        


# In[71]:


model_data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'total_words': total_words,
    'tags': tags    
}


# In[72]:


FILE = 'model_data.pth'
torch.save(model_data,FILE)
print(f'Training has been successful and model has been saved to file {FILE}')


# In[ ]:




