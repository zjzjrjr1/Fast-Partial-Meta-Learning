# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:55:58 2024

@author: ra064640
"""

import torch 
import gc
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import random

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



def learning_transfer(meta_model, model, transfer_ratio):
    # first see how many parameters there are in the model 
    i = 0
    for name, param in meta_model.named_parameters():
        i += 1
    transfer_layer_idx = int(round(i * transfer_ratio))

    i = 0
    
    # the parameter after the transfer_ratio are set to False
    for param_first, param_second in zip(meta_model.parameters(), model.parameters()):
        # transfer the weights 
        if i > transfer_layer_idx:
            # then transfer the parameter
            param_second.copy_ = param_first
            # then freeze the parameter
            param_second.requires_grad = False
        else:
            pass
        i += 1
    return model



def grad_to_true(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def meta_dataloader(dataset, batch_size, sample_ratio):
    dataset_len = len(dataset)
    sample_length = int(dataset_len * sample_ratio)
    random_list = random.sample(range(1,dataset_len), sample_length)
    meta_dataset = Subset(dataset, random_list)
    return DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)

def accuracy(outputs, labels):
    # model output is the size of batch x label
    # get the labels from the outputs. 
    batch_size = outputs.shape[0]
    val, idx = torch.max(outputs, dim = -1)
    correct = idx.eq(labels).sum() * 1.0
    acc = correct / batch_size
    return acc

def meta_train_epoch(meta_epoch,meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer):
    meta_epoch_running_loss = 0
    for meta_train_idx, (features, labels) in enumerate(meta_train_dataloader):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            meta_model = meta_model.cuda()
        
        meta_optimizer.zero_grad()
        
        outputs = meta_model(features)
        loss = meta_loss_func(outputs, labels)
        loss.backward()
        meta_optimizer.step()
        
        meta_epoch_running_loss += loss.item()
        if meta_train_idx % 50 == 0:
            print(f'the meta training batch loss at meta epoch of  {meta_epoch} idx of {meta_train_idx} is {loss}')
    return meta_model, meta_epoch_running_loss



def train_epoch(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer):
    
    epoch_running_loss = 0
    for train_idx, (features, labels) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            model = model.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_running_loss += loss.item()
        if train_idx % 50 ==  0:
            print(f'the training batch loss at meta epoch {meta_epoch} epoch {epoch} idx of {train_idx} is {loss}')
    return model, epoch_running_loss

def valid_epoch(model, valid_dataloader):
    
    valid_batch_acc_list = []
    for valid_idx, (features, labels) in enumerate(valid_dataloader):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            model = model.cuda()
            
        with torch.no_grad():
            output = model(features)
        batch_acc = accuracy(output, labels) 
        
        valid_batch_acc_list.append(batch_acc.item())
        
    return valid_batch_acc_list
    
    
########################## start the actual code from here ##########################
#####################################################################################
all_epochs = 4

meta_epochs = 2
train_epochs = 1

batch_size = 128
sample_ratio = 0.25
transfer_ratio = 0.4

model = models.resnet18(weights=None)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


meta_loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss()




for all_epoch in range(all_epochs):
    
    # before the meta_train, every parameter needs to have the require grad to True
    meta_model = grad_to_true(model)
    # since a new model is made, need to reset the new optimizer with the new model
    meta_optimizer = torch.optim.Adam(meta_model.parameters())
    meta_model.train()
    meta_train_dataloader = meta_dataloader(training_data, batch_size, sample_ratio)
    for meta_epoch in range(meta_epochs): 
        meta_model, meta_epoch_loss = meta_train_epoch(meta_epoch, meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer)
    
    # do model transfer.
    # the model should have : the later layers = require_grad = False, first layers = require_grad
    # take the meta_learner and replace the weights in the original model
    model = learning_transfer(meta_model, model, transfer_ratio)
    # since the model is made again, need to initialize the optimizer again. 
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(train_epochs):
        model, train_epoch_loss = train_epoch(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer)
    
    # run the validation function to see the accuracy of the epoch and batches!
    model.eval()
    with torch.no_grad():
        valid_acc_list = valid_epoch(model, test_dataloader)
    # valid acc list average
    valid_ave = sum(valid_acc_list)/len(valid_acc_list)
    print(f'validation acc for the meta epoch {meta_epoch} is {valid_ave}')
    
    # implement memory clear 
    gc.collect()
    torch.cuda.empty_cache()

# computing amount calculation
print(meta_epochs*sample_ratio + meta_epochs * train_epochs * transfer_ratio)
###################################################################################