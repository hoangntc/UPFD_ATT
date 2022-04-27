import os, sys, re, datetime, random, gzip, json
import commentjson
from collections import OrderedDict
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import torch
import torch.nn as nn

def read_json(fname):
    '''
    Read in the json file specified by 'fname'
    '''
    with open(fname, 'rt') as handle:
        return commentjson.load(handle, object_hook=OrderedDict)

def metrics(logits, labels):
    # preds = torch.round(torch.cat(preds)).cpu().detach()
    # gts = torch.cat(gts).cpu().detach()
    preds = torch.round(logits).cpu().detach()
    gts = labels.cpu().detach()
    acc = accuracy_score(preds, gts)
    f1 = f1_score(preds, gts)
    return torch.tensor([acc]), torch.tensor([f1])