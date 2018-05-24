# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: Dict.py

@time: 18-5-18 下午2:10

@desc:
'''
from utils import Constants
import torch


class Dict(object):

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.id2word = {idx: word for word, idx in word2idx.items()}

    def getIdx(self, word):
        return self.word2idx.get(word, Constants.UNK)

    def getWord(self, idx):
        return self.id2word.get(idx, Constants.UNK_WORD)

    def convert2idx(self, words):
        vec = [self.getIdx(word) for word in words]

        return torch.LongTensor(vec)

    def convert2word(self, idxs):
        words = [self.getWord(idx) for idx in idxs]

        return words

    def size(self):
        return len(self.word2idx)