# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: train.py

@time: 18-5-24 上午11:42

@desc:
'''

import json
import torch
import torch.optim as optim
from torch.autograd import Variable


from torch.utils.data import DataLoader
from utils.process import KWDataSet
from utils.AoAKW import AoAKW

def train(train_loader, test_loader):

    model = AoAKW(vocab_dict, dropout_rate=0.3, embed_dim=50, hidden_dim=50, n_class=92)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(5):
        for batch_idx, samples in enumerate(train_loader, 0):
            v_docs, v_titles, v_kws, v_topics = tansform(samples)

            topic_probs, kw_probs = model(v_docs, v_titles, v_kws, v_topics)
            optimizer.zero_grad()
            loss = None # todo
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                # log loss
                pass
            if batch_idx % 100 == 0:
                evalution(model, test_loader)


def evalution(model, dataloader):
    model.eval()
    for batch_idx, samples in enumerate(dataloader, 0):
        v_docs, v_titles, v_kws, v_topics = tansform(samples)
        topic_probs, kw_probs = model(v_docs, v_titles, v_kws, v_topics)
        loss = None # todo

    # log loss



if __name__ =='__main__':
    trainset = KWDataSet()
    train_loader = DataLoader(trainset, batch_size=48, shuffle=True, num_workers=4)

    testset = KWDataSet()
    test_loader = DataLoader(testset, batch_size=48, shuffle=True, num_workers=4)

    train(train_loader, test_loader)