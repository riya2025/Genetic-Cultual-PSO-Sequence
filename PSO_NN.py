#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import tensorflow as tf


# In[18]:


data = pd.read_csv(r"C:\Users\LEN\Downloads\new_test (1).csv")
data.drop(["contact","month","day_of_week","duration","campaign","poutcome"] , axis = 1,inplace=True)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
x = torch.tensor(x , dtype = torch.float64)
y = torch.tensor(y , dtype=  torch.float64)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25)
    


# In[39]:


class nn( ):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(7, 20 )
        self.linear2 = torch.nn.Linear(20, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.linear3(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x


# In[40]:


model = nn()
loss_function = torch.nn.CrossEntropyLoss()


# In[41]:


class PSO:
    def __init__(self, model, w, c1, c2, num_of_particles, decay , inputs, labels):
        self.model = model
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_of_particles = num_of_particles
        self.inputs = inputs
        self.labels = labels
        self.initialize_position()
        self.initialize_velocity()
        self.pbest = self.positions
        self.gbest = np.inf
        self.decay = decay
        
    def initialize_position(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r1 = np.random.rand(self.num_of_particles, num_params)
        self.positions = (10*r1)-0.5
        
    def initialize_velocity(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r2 = np.random.rand(self.num_of_particles, num_params)
        self.velocity = r2 - 0.5
        
    def find_pbest(self):
        for i in range(len(self.pbest)):
            if self.fitness(self.pbest[i]) > self.fitness(self.positions[i]):
                self.pbest[i] = self.positions[i]
                
    def find_gbest(self):
        for position in self.positions:
            if self.fitness(position) < self.fitness(self.gbest):
                self.gbest = position
                
    def new_velocity(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r1 = np.random.rand(self.num_of_particles, num_params)
        r2 = np.random.rand(self.num_of_particles, num_params)
        self.velocity = (self.w*self.velocity) + (self.c1 * r1 * (self.pbest - self.positions)) + (self.c2 * r2  * (self.gbest - self.positions))
    
    def new_position(self):
        self.positions += self.velocity
        
    def fitness(self, weights):
        
        outputs = self.model(self.inputs.float())
        loss = torch.nn.functional.binary_cross_entropy(outputs.float(), self.labels.reshape([len(self.inputs.float()) , 1]).float())
        return loss.item()
    
    def update_weights(self):
        self.find_pbest()
        self.find_gbest()
        self.new_velocity()
        self.new_position()
        fitness_scores = [self.fitness(weights) for weights in self.positions]
        best_index = np.argmin(fitness_scores)
        best_weights = self.positions[best_index]
    def decay_w(self):
        self.w = self.w - (self.w*self.decay)
    


# In[44]:


model = nn.Sequential(nn.Linear(x_train.shape[1], 1),nn.Sigmoid())


# In[45]:


pso= PSO(model, w = 0.8 , c1 = 0.1 , c2 = 0.1 , num_of_particles =20 , decay = 0.03, inputs = x_train, labels = y_train)


# In[46]:


n=10
loss_list= []
for epoch in range(n):
            pso.update_weights()
            outputs = model(x_train.float())
            loss = torch.nn.functional.binary_cross_entropy(outputs.float(), y_train.reshape([len(x_train), 1]).float())
            loss_list.append(loss.item())
            loss.backward()
            if epoch % 10 == 0:
                print("Epoch", epoch, ": ", loss.item())
                pso.decay_w()
print(loss_list)  


# In[ ]:




