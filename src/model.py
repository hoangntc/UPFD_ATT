import os, sys, re, datetime, random, gzip, json, copy, tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
from time import time
from math import ceil

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from torch_geometric.nn import Sequential, HeteroConv, GINConv, GCNConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.nn import global_max_pool as gmp

PROJ_PATH = Path(os.path.join(re.sub("/UPFD_ATT.*$", '', os.getcwd()), 'UPFD_ATT'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
import utils
from utils import *

class GNN(pl.LightningModule):
    '''
    GNN-based fake new detector
    '''
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # Graph Convolutions
        self.conv1 = GATConv(self.hparams.in_channels, self.hparams.hidden_channels)
        self.conv2 = GATConv(self.hparams.hidden_channels, self.hparams.hidden_channels)
        self.conv3 = GATConv(self.hparams.hidden_channels, self.hparams.hidden_channels)
        
        # Readout
        self.lin_news = Linear(self.hparams.in_channels, self.hparams.hidden_channels)
        self.lin0 = Linear(self.hparams.hidden_channels, self.hparams.hidden_channels)
        self.lin1 = Linear(2*self.hparams.hidden_channels, self.hparams.out_channels)

        # loss
        self.loss = torch.nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer  
    
    def forward(self, x, edge_index, batch):
        # Graph Convolutions
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        # Pooling
        h = gmp(h, batch)

        # Readout
        h = self.lin0(h).relu()

        # According to UPFD paper: Include raw word2vec embeddings of news 
        # This is done per graph in the batch
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        # root is e.g. [   0,   14,   94,  171,  230,  302, ... ]
        news = x[root]
        news = self.lin_news(news).relu()
        
        out = self.lin1(torch.cat([h, news], dim=-1))
        logits = torch.sigmoid(out)
        return logits
        
    def training_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)        
        return ce_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)   
        acc, f1 = metrics(logits, labels)

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'f1': f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean().cpu()
        avg_f1 = sum([x['f1'] for x in val_step_outputs])/ len([x['f1'] for x in val_step_outputs])
        logs = {
            'val_loss': avg_loss, 
            'val_acc': avg_acc,
            'val_f1': avg_f1,
        }
        self.log_dict(logs, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)   
        acc, f1 = metrics(logits, labels)

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'f1': f1,
        }
        return logs
    
    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in test_step_outputs]).mean().cpu()
        avg_f1 = sum([x['f1'] for x in test_step_outputs])/ len([x['f1'] for x in test_step_outputs])
        logs = {
            'test_loss': avg_loss, 
            'test_acc': avg_acc,
            'test_f1': avg_f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs

class GNN_ATT(pl.LightningModule):
    '''
    GNN-based with attention fake new detector
    '''
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # Graph Convolutions
        self.conv1 = GATConv(self.hparams.in_channels, self.hparams.hidden_channels)
        self.conv2 = GATConv(self.hparams.hidden_channels, self.hparams.hidden_channels)
        self.conv3 = GATConv(self.hparams.hidden_channels, self.hparams.hidden_channels)
        
        # Att
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.n_head = self.hparams.n_head
        self.w = Parameter(torch.Tensor(self.hparams.n_head, self.hparams.hidden_channels, self.hparams.hidden_channels))
        self.a_src = Parameter(torch.Tensor(self.hparams.n_head, self.hparams.hidden_channels, 1))
        self.a_dst = Parameter(torch.Tensor(self.hparams.n_head, self.hparams.hidden_channels, 1))
        self.reset_parameters()
        
        # Readout
        self.lin_news = Linear(self.hparams.in_channels, self.hparams.hidden_channels)
        self.lin0 = Linear(self.hparams.hidden_channels, self.hparams.hidden_channels)
        self.lin1 = Linear((self.hparams.n_head+1)*self.hparams.hidden_channels, self.hparams.out_channels)

        # loss
        self.loss = torch.nn.BCELoss()

    def reset_parameters(self):
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer  
    
    def reshape(self, x_out, batch):
        batch_size = int(batch.max().item() + 1)
        tensors = []
        max_size = 0
        for i in range(batch_size):
            idx = (batch == i).nonzero(as_tuple=True)[0].cpu().numpy()
            tensors.append(x_out[idx, :])
            if len(idx) > max_size: max_size = len(idx)
        
        padding_tensors = []
        for tensor in tensors:
            m = nn.ZeroPad2d((0, 0, max_size - tensor.shape[0], 0))
            padding_tensors.append(m(tensor))
    
        return torch.stack(padding_tensors)

    def forward(self, x, edge_index, batch):
        # Graph Convolutions
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        
        # The reference paper use pooling
        # We need to reshape the output for attention (batch_size, max_no_node, hidden_channels)
        # h = gmp(h, batch)
        reshape_h = self.reshape(h, batch)
    
        # According to UPFD paper: Include raw word2vec embeddings of news 
        # This is done per graph in the batch
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1) # lag to get index of root
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        # root is e.g. [   0,   14,   94,  171,  230,  302, ... ]
        news = x[root]
        news = self.lin_news(news).relu()
        
        # Attention
        bs, n = reshape_h.size()[:2]
        h_prime = torch.matmul(reshape_h.unsqueeze(1), self.w) # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn) # bs x n_head x n x n
        h = torch.matmul(attn, h_prime)
        h = h.mean(dim=2)
        h = torch.flatten(h, start_dim=1, end_dim=2)
        
        # Feed-forward
        out = torch.cat([h, news], dim=-1)
        out = self.lin1(out)
        logits = torch.sigmoid(out)
        return logits
        
    def training_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)        
        return ce_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)   
        acc, f1 = metrics(logits, labels)

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'f1': f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean().cpu()
        avg_f1 = sum([x['f1'] for x in val_step_outputs])/ len([x['f1'] for x in val_step_outputs])
        logs = {
            'val_loss': avg_loss, 
            'val_acc': avg_acc,
            'val_f1': avg_f1,
        }
        self.log_dict(logs, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        logits = self.forward(
            x=batch['x'],
            edge_index=batch['edge_index'],
            batch=batch['batch'],
        )
        
        logits = torch.reshape(logits, (-1,))
        labels = batch['y'].float()
        ce_loss = self.loss(logits, labels)   
        acc, f1 = metrics(logits, labels)

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'f1': f1,
        }
        return logs
    
    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in test_step_outputs]).mean().cpu()
        avg_f1 = sum([x['f1'] for x in test_step_outputs])/ len([x['f1'] for x in test_step_outputs])
        logs = {
            'test_loss': avg_loss, 
            'test_acc': avg_acc,
            'test_f1': avg_f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs
