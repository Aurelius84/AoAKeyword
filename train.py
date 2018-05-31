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
import shutil
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

time_stamp = time.strftime("%m-%d-%H:%M:%S", time.localtime())
vis = Visualizer(log_dir="runs/%s"%time_stamp)
criterion = nn.CrossEntropyLoss()

def train(train_loader, test_loader):

    model = AoAKW(word2idx, dropout_rate=0.3, embed_dim=50, hidden_dim=50, n_class=92)
    lr = 1e-4
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    best_kw_acc = 0.
    for epoch in range(10):
        lg_loss, cr_loss, ls = 0, 0, 0
        for batch_idx, samples in enumerate(train_loader, 0):
            v_docs, v_titles, v_kws, v_topics = process.transform(samples, return_variable=True)

            topic_probs, kw_probs, atten_s = model(v_docs, v_titles, v_kws, v_topics)
            optimizer.zero_grad()

            log_loss = - torch.mean(torch.log(kw_probs), dim=0)
            crossEntro = criterion(topic_probs, v_topics)
            loss = 1.4 * log_loss + 0.6 * crossEntro
            loss.backward()
            optimizer.step()

            lg_loss += log_loss.data[0]
            cr_loss += crossEntro.data[0]
            ls += loss.data[0]

            if batch_idx % 10 == 0:
                # log loss
                vis.plot('train/log_loss', lg_loss/10)
                vis.plot('train/crossEntro', cr_loss/10)
                vis.plot('train/loss', loss/10)
                lg_loss, cr_loss, ls = 0, 0, 0

                topic_pre_num, topic_gt_num, kw_num_correct, kw_gt_num = acc_num(v_kws, atten_s, v_topics, topic_probs, v_docs)
                vis.plot("train/topic_acc", float(topic_pre_num)/topic_gt_num)
                vis.plot("train/kw_acc", float(kw_num_correct)/kw_gt_num)

            if batch_idx % 100 == 0:
                kw_acc = evalution(model, test_loader)
                checkPredict(atten_s, v_docs, v_kws, topic_probs, v_topics)

                if kw_acc > best_kw_acc:
                    best_kw_acc = kw_acc
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_kw_acc': best_kw_acc,
                        'optimizer': optimizer.state_dict(),
                    }, False)

                model.train()


def acc_num(v_kws, atten_s, v_topics, topic_prob, v_docs):
    topic_num_correct = v_topics.data == np.argmax(topic_prob.data, axis=1)
    topic_pre_num = torch.sum(topic_num_correct)
    topic_gt_num = float(v_topics.size(0))

    # top 3
    topk, indices = torch.topk(atten_s, 3, dim=1)
    indices_ = indices.data.view(-1, 3).numpy()
    v_docs_ = v_docs.data.numpy()
    kw_num_correct = 0
    kw_gt_num = 0

    for i in range(len(v_docs_)):
        gt_kws = v_kws[i].data.numpy()
        pre_kws = v_docs_[i][indices_[i]]
        kw_num_correct += len(set(gt_kws) & set(pre_kws))
        kw_gt_num += sum(gt_kws != 0)

    return topic_pre_num, topic_gt_num, kw_num_correct, kw_gt_num


def evalution(model, dataloader):
    model.eval()
    lg_loss, cr_loss, ls = 0, 0, 0
    topic_pre_num, topic_gt_num, kw_num_correct, kw_gt_num = 0.,0.,0.,0.
    for batch_idx, samples in enumerate(dataloader, 0):
        v_docs, v_titles, v_kws, v_topics = process.transform(samples)
        topic_probs, kw_probs, atten_s = model(v_docs, v_titles, v_kws, v_topics)

        log_loss = - torch.mean(torch.log(kw_probs), dim=0)
        crossEntro = criterion(topic_probs, v_topics)
        loss = log_loss + crossEntro

        lg_loss += log_loss.data[0]
        cr_loss += crossEntro.data[0]
        ls += loss.data[0]

        tp_num, tg_num, kp_num, kg_num = acc_num(v_kws, atten_s, v_topics, topic_probs, v_docs)
        topic_pre_num += tp_num
        topic_gt_num += tg_num
        kw_num_correct += kp_num
        kw_gt_num += kg_num

    vis.plot('test/log_loss', lg_loss / batch_idx)
    vis.plot('test/crossEntro', cr_loss / batch_idx)
    vis.plot('test/loss', ls / batch_idx)
    vis.plot("test/topic_acc", topic_pre_num/topic_gt_num)
    vis.plot("test/kw_acc", kw_num_correct/kw_gt_num)

    return kw_num_correct/kw_gt_num


def save_checkpoint(state, is_best, filename='model/%s_checkpoint.pth.tar' % time_stamp):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model/%s_model_best.pth.tar' % time_stamp)


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
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    train(train_loader, test_loader)