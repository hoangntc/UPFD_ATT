import os, sys, re, datetime, random, gzip, json, copy
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
from time import time
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything


PROJ_PATH = Path(os.path.join(re.sub("/UPFD_ATT.*$", '', os.getcwd()), 'UPFD_ATT'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))


class DataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = UPFD(
                root=self.hparams.root, 
                name=self.hparams.name, 
                feature=self.hparams.feature, 
                split="train")
            self.data_val = UPFD(
                root=self.hparams.root, 
                name=self.hparams.name, 
                feature=self.hparams.feature, 
                split="val")
            
        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            self.data_test = UPFD(
                root=self.hparams.root, 
                name=self.hparams.name, 
                feature=self.hparams.feature, 
                split="test")
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=1, 
            shuffle=True,
            drop_last=True, 
        )
    
    def mytrain_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=1, 
            shuffle=False,
            drop_last=False, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=1, 
            shuffle=False,
        )
    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=1, 
            shuffle=False,
        )