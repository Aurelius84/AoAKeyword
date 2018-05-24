# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: visualize.py

@time: 18-5-24 下午5:25

@desc:
'''

from tensorboard_logger import Logger


class Visualizer():
    def __init__(self, log_dir='runs/', **kwargs):
        self.tenbd = Logger(log_dir, flush_secs=10)

        self.index = {}

        self.log_text = ''

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.tenbd.log_value(name, y, x)
        self.index[name] = x+1

    def plotMany(self, data):
        for k, v in data.iteritems():
            self.plot(k,v)