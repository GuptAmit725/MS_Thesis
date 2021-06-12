#!/usr/bin/env python
# coding: utf-8

# In[14]:


import json
import random
import torch
import torch.nn as nn
from model import BotNet
from NLP_utils import  tokenize, vectorize


# In[18]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


with open('intents.json') as f:
    intents = json.load(f)
print(intents)


# In[5]:


model_data = torch.load('model_data.pth')


# In[6]:


model_data


# In[8]:


model_data.keys()


# In[9]:


model_state = model_data['model_state']
input_size = model_data['input_size']
output_size = model_data['output_size']
hidden_size = model_data['hidden_size']
total_words = model_data['total_words']
tags = model_data['tags']


# In[13]:


bot_model = BotNet(input_size, hidden_size, output_size)
bot_model.load_state_dict(model_state)
bot_model.eval()


# In[23]:


bot_name = 'test_bot'
print(f'Ask anything!and type quit to exit.')

while True:
    sent = input('You: ')
    if sent == 'quit':
        break
    sent_tok = tokenize(sent)
    sent_vec = vectorize(sent_tok, total_words)
    sent_vec =sent_vec.reshape(1,-1)
    sent_vec = torch.from_numpy(sent_vec).to(device).float()
    
    out = bot_model(sent_vec)
    prob_of_out = torch.softmax(out,dim=1)
    _,pred = torch.max(out,dim=1)
    prob = prob_of_out[0][pred.item()]
    #print(prob.item(),prob_of_out[0])
    if prob.item() > 0.2:
        for intent in intents['intents']:
            if tags[pred.item()] == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
                
    else:
        print(f'{bot_name}: I do not understand...')
    
    
    
        


# In[ ]:




