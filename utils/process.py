# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: process.py

@time: 18-5-24 上午11:30

@desc:
'''

import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.Dict import Dict


class KWDataSet(Dataset):
    def __init__(self, file_path, voc_path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class KWSample(object):
    def __init__(self, doc, title, kws=None, topic=None):
        self.doc = doc
        self.title = title
        self.kws = kws
        self.topic = topic

        # vector placeholder


class Process(object):
    def __init__(self, word2idx):
        self.dict = Dict(word2idx)

    def transform(self, sample):
        pass