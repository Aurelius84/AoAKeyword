# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: train.py

@time: 18-5-24 上午11:42

@desc:
'''
import time
import json
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import OrderedDict

from torch.utils.data import DataLoader
from utils.process import KWDataSet
from utils.AoAKW import AoAKW
from utils.process import Process
from utils.Dict import Dict
from utils.visualize import Visualizer

word2idx = Dict(json.load(open('docs/word2idx.json', 'r')))
cate2idx = Dict(json.load(open('docs/cate2idx.json', 'r')))
process = Process(word2idx, cate2idx)

vis = Visualizer(log_dir="runs/%s"%time.strftime("%m-%d-%H:%M:%S", time.localtime()))
criterion = nn.CrossEntropyLoss()

def train(train_loader, test_loader):

    model = AoAKW(word2idx, dropout_rate=0.3, embed_dim=50, hidden_dim=50, n_class=92)
    lr = 1e-4
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(5):
        lg_loss, cr_loss, ls = 0, 0, 0
        for batch_idx, samples in enumerate(train_loader, 0):
            v_docs, v_titles, v_kws, v_topics = process.transform(samples, return_variable=True)

            topic_probs, kw_probs, atten_s = model(v_docs, v_titles, v_kws, v_topics)
            optimizer.zero_grad()

            log_loss = - torch.mean(torch.log(kw_probs), dim=0)
            crossEntro = criterion(topic_probs, v_topics)
            loss = log_loss + crossEntro
            loss.backward()
            optimizer.step()

            lg_loss += log_loss.data[0]
            cr_loss += crossEntro.data[0]
            ls += loss.data[0]

            if batch_idx % 5 == 0:
                # log loss
                vis.plot('train/log_loss', lg_loss/10)
                vis.plot('train/crossEntro', cr_loss/10)
                vis.plot('train/loss', loss/10)
                lg_loss, cr_loss, ls = 0, 0, 0

                topic_acc, kw_acc = accuracy(v_kws, atten_s, v_topics, topic_probs, v_docs)
                vis.plot("train/topic_acc", topic_acc)
                vis.plot("train/kw_acc", kw_acc)

            if batch_idx % 6 == 0:
                evalution(model, test_loader)
                checkPredict(atten_s, v_docs, v_kws, topic_probs, v_topics)
                model.train()


def accuracy(v_kws, atten_s, v_topics, topic_prob, v_docs):
    topic_num_correct = v_topics.data == np.argmax(topic_prob.data, axis=1)
    topic_acc = torch.sum(topic_num_correct) / float(v_topics.size(0))

    # top 3
    topk, indices = torch.topk(atten_s, 3, dim=1)
    indices_ = indices.data.view(-1, 3).numpy()
    v_docs_ = v_docs.data.numpy()
    kw_num_correct = 0

    for i in range(len(v_docs_)):
        gt_kws = v_kws[i].data.numpy()
        pre_kws = v_docs_[i][indices_[i]]
        kw_num_correct += len(set(gt_kws) & set(pre_kws))

    kw_acc = kw_num_correct / float(3*len(v_docs_))

    return topic_acc, kw_acc


def evalution(model, dataloader):
    model.eval()
    lg_loss, cr_loss, ls = 0, 0, 0
    t_acc, k_acc = 0, 0
    for batch_idx, samples in enumerate(dataloader, 0):
        v_docs, v_titles, v_kws, v_topics = process.transform(samples)
        topic_probs, kw_probs, atten_s = model(v_docs, v_titles, v_kws, v_topics)

        log_loss = - torch.mean(torch.log(kw_probs), dim=0)
        crossEntro = criterion(topic_probs, v_topics)
        loss = log_loss + crossEntro

        lg_loss += log_loss.data[0]
        cr_loss += crossEntro.data[0]
        ls += loss.data[0]

        topic_acc, kw_acc = accuracy(v_kws, atten_s, v_topics, topic_probs, v_docs)
        t_acc+= topic_acc
        k_acc += kw_acc

    vis.plot('test/log_loss', lg_loss / batch_idx)
    vis.plot('test/crossEntro', cr_loss / batch_idx)
    vis.plot('test/loss', ls / batch_idx)
    vis.plot("test/topic_acc", t_acc/batch_idx)
    vis.plot("test/kw_acc", k_acc/batch_idx)


def checkPredict(atten_s, v_docs, v_kws, topic_probs, v_topics, k=5):
    topic_pred = np.argmax(topic_probs.data, axis=1)

    # top 3
    topk, indices = torch.topk(atten_s, 3, dim=1)
    indices_ = indices.data.view(-1, 3).numpy()
    v_docs_ = v_docs.data.numpy()
    gt_kws = v_kws.data.numpy()
    gt_topics = v_topics.data.numpy()
    pre_kws = []
    for i in range(len(v_docs_)):
        pre_kws.append(v_docs_[i][indices_[i]])
    for i in range(k):
        data = OrderedDict({
            'topic': cate2idx.getWord(gt_topics[i]),
            'pre_topic': cate2idx.getWord(topic_pred[i]),
            'kws': " ".join(word2idx.convert2word(gt_kws[i])),
            'pre_kws': " ".join(word2idx.convert2word(pre_kws[i])),
            'docs': " ".join(word2idx.convert2word(v_docs_[i]))
        })
        print(json.dumps(data, ensure_ascii=False, indent=4))


if __name__ =='__main__':
    trainset = KWDataSet('docs/train.txt')
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = KWDataSet('docs/test.txt')
    test_loader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

    train(train_loader, test_loader)