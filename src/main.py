import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

PROJ_PATH = Path(os.path.join(re.sub("/UPFD_ATT.*$", '', os.getcwd()), 'UPFD_ATT'))
sys.path.insert(1, str(PROJ_PATH / 'src'))
import utils
from dataset import DataModule
from model import GNN, GNN_ATT
from trainer import build_trainer

def main():
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('-config_file', help='config file path', default=str(PROJ_PATH / 'src/config/gnn_att_gos.json'), type=str)
    args = parser.parse_args()
    print('Reading config file:', args.config_file)
    args.config = utils.read_json(args.config_file)
    seed_everything(args.config['trainer_params']['seed'], workers=True)
    
    data_module = DataModule(args.config['data_params'])
    if args.config['model_params']['model_name'] == 'GNN':
        model_module = GNN(args.config['model_params'])
    elif args.config['model_params']['model_name'] == 'GNN_ATT':
        model_module = GNN_ATT(args.config['model_params'])
    trainer, _ = build_trainer(args.config['trainer_params'])

    # Train
    print('### Train')
    trainer.fit(model_module, data_module)
    
    # Test
    print('### Test')
    checkpoint_dir = Path(args.config['trainer_params']['checkpoint_dir'])
    print(f'Load checkpoint from: {str(checkpoint_dir)}')
    paths = sorted(checkpoint_dir.glob('*.ckpt'))
    name = args.config['trainer_params']['name']
    filtered_paths = [p for p in paths if f'model={name}-' in str(p)]
    results = []
    for i, p in enumerate(filtered_paths):
        print(f'Load model {i}: {p}')
        # test
        model_test = model_module.load_from_checkpoint(checkpoint_path=p) 
        result = trainer.test(model_test, datamodule=data_module)
        results.append(results)
        del model_test

    for p, r in zip(filtered_paths, results):
        print(p)
        print(r)
        print('\n')

    del data_module
    del model_module
    del trainer
    
if __name__ == "__main__":
    main()