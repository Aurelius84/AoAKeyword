# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: process.py

@time: 18-5-24 上午11:30

@desc:
'''
import json
import torch
import mmap
import numpy as np
from tqdm import tqdm
from utils import Constants
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils.Dict import Dict


class KWDataSet(Dataset):
    def __init__(self, file_path):
        self.data = open(file_path, 'r').readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.data[idx].strip("\n").split("\t")
        kws = items[4][1:-1].replace("\"", "")
        return KWSample(items[-1], items[-1], kws, items[0]).__dict__


class KWSample(object):
    def __init__(self, doc, title, kws=None, topic=None):
        self.doc = doc
        self.title = title
        self.kws = kws
        self.topic = topic

        # vector placeholder


class Process(object):
    def __init__(self, word2idx, cate2idx, cuda=True):
        self.dict = word2idx
        self.cate = cate2idx
        self.cuda = torch.cuda.is_available() if cuda else False

    def transform(self, samples, return_variable=True):
        docs = [doc.split() for doc in samples['doc']]
        v_docs = self.toTensor(docs)

        titles = [title.split() for title in samples['title']]
        v_titles = self.toTensor(titles)

        v_kws, v_topics = torch.zeros([1]),torch.zeros([1])
        if samples['kws'] is not None:
            kws = [kw.split(",") for kw in samples['kws']]
            v_kws = self.toTensor(kws)

        if samples['topic'] is not None:
            v_topics = torch.LongTensor([self.cate.getIdx(lbl) for lbl in samples['topic']])

        if return_variable:
            v_docs, v_titles,v_kws,v_topics = Variable(v_docs), Variable(v_titles),Variable(v_kws),Variable(v_topics)

        return self.wrapper(v_docs), self.wrapper(v_titles), self.wrapper(v_kws), self.wrapper(v_topics)

    def pad(self, input):
        max_len = max([len(seq) for seq in input])
        output = [seq + [Constants.PAD_WORD] * (max_len - len(seq)) for seq in input]

        return output

    def toTensor(self, input):
        input = self.pad(input)
        long_tensor = torch.stack([self.dict.convert2idx(seq) for seq in input], dim=0)
        return long_tensor

    def onehot(self, label):
        output = np.zeros((len(label), self.cate.size()))
        for i, lbl in enumerate(label):
            output[i][self.cate.getIdx(lbl)] = 1

        return torch.LongTensor(output)

    def wrapper(self, input):
        return input.cuda() if self.cuda else input


def build_vocab(file_path):
    vocab = set()
    cate = []
    with open(file_path,'r') as f:
        for line in tqdm(f, total=get_num_lines(file_path)):
            items = line.strip("\n").split("\t")
            vocab |= set(items[-1].split())
            cate.append(items[0])

        word2idx = dict(zip(vocab, range(2, len(vocab)+2)))
        word2idx[Constants.PAD_WORD] = Constants.PAD
        word2idx[Constants.UNK_WORD] = Constants.UNK

        cate = set(cate)
        cate2idx = dict(zip(cate, range(len(cate))))

    with open('../docs/word2idx.json', 'w') as fw, open('../docs/cate2idx.json', 'w') as fc:
        word2idx = json.dumps(word2idx, ensure_ascii=False, indent=4)
        fw.write(word2idx)

        cate2idx = json.dumps(cate2idx, ensure_ascii=False, indent=4)
        fc.write(cate2idx)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    fp.close()
    return lines


if __name__ == '__main__':
    build_vocab('../docs/0426-topic-kw-ts.txt')
    exit()
    dataset = KWDataSet('../docs/test.txt')
    word2idx = json.load(open('../docs/word2idx.json', 'r'))
    cate2idx = json.load(open('../docs/cate2idx.json', 'r'))
    process = Process(word2idx,cate2idx)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=6, num_workers=1)
    for i, samples in enumerate(dataloader, 0):
        print(samples['title'])
        v_docs, v_titles, v_kws, v_topics = process.transform(samples)
        print(v_topics, v_kws,)
        exit()

