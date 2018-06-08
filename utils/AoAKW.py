# coding: utf-8

'''
@author: zhangliujie

@contact: zhangliujie@xiaomi.com

@file: AoAKW.py

@time: 18-5-18 下午2:19

@desc:
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import torch.nn.init as weight_init

from utils import Constants


def sort_batch(data, seq_len):
    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx]
    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)
    return sorted_data, sorted_seq_len, reverse_idx


def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    shift, _ = torch.max(input, axis, keepdim=True)
    shift = shift.expand_as(input)

    target_exp = torch.exp(input-shift)*mask.float()
    normalize = torch.sum(target_exp, axis, keepdim=True).expand_as(target_exp)
    softm = target_exp / (normalize + epsilon)

    return softm

def get_seq_lenth(input, return_variable=True):
    seq_len = torch.LongTensor([torch.nonzero(input.data[i]).size(0) for i in range(input.data.size(0))])
    if return_variable:
        seq_len = Variable(seq_len, requires_grad=True)

    return seq_len


def creat_mask(seq_lens, return_variable=True):
    mask = torch.zeros(seq_lens.data.size(0), torch.max(seq_lens.data))
    for i, seq_len in enumerate(seq_lens.data):
        mask[i][:seq_len] = 1
    if return_variable:
        mask = Variable(mask, requires_grad=True)
    return cuda_wrapper(mask)


def cuda_wrapper(input):
    return input.cuda() if torch.cuda.is_available() else input


class AoAKW(nn.Module):

    def __init__(self, vocab_dict, dropout_rate, embed_dim, hidden_dim, n_class, bidirectional=True):
        super(AoAKW, self).__init__()
        self.vocab_dict = vocab_dict
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.n_class = n_class

        self.embedding = nn.Embedding(vocab_dict.size(), self.embed_dim, padding_idx=Constants.PAD)
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        input_size = self.embed_dim
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_dim, dropout=dropout_rate,
                          bidirectional=bidirectional, batch_first=True)

        for weight in self.gru.parameters():
            if len(weight.size()) > 1:
                weight_init.orthogonal(weight.data)

        self.fc = nn.Sequential(
            nn.Linear((int(bidirectional)+1)*self.hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_class)
        )

    def forward(self, docs_input, titles_input, keywords=None, topics=None):
        docs_len, titles_len = get_seq_lenth(docs_input), get_seq_lenth(titles_input)
        docs_mask, titles_mask = creat_mask(docs_len), creat_mask(titles_len)

        s_docs, s_docs_len, reverse_docs_idx = sort_batch(docs_input, docs_len)
        s_titles, s_titles_len, reverse_titles_idx = sort_batch(titles_input, titles_len)

        # embedding
        docs_embedding = pack(self.embedding(s_docs), list(s_docs_len.data), batch_first=True)
        titles_embedding = pack(self.embedding(s_titles), list(s_titles_len.data), batch_first=True)

        # GRU encoder
        docs_outputs, _ = self.gru(docs_embedding, None)
        titles_outputs, _ = self.gru(titles_embedding, None)

        # unpack
        docs_outputs, _ = unpack(docs_outputs, batch_first=True)
        titles_outputs, _ = unpack(titles_outputs, batch_first=True)

        # unsort
        docs_outputs = docs_outputs[reverse_docs_idx]
        titles_outputs = titles_outputs[reverse_titles_idx]

        # calculate attention matrix
        dos = docs_outputs
        doc_mask = docs_mask.unsqueeze(2)
        tos = torch.transpose(titles_outputs, 1, 2)
        title_mask = titles_mask.unsqueeze(2)

        M = torch.bmm(dos, tos)
        M_mask = torch.bmm(doc_mask, title_mask.transpose(1, 2))
        alpha = softmax_mask(M, M_mask, axis=1)
        beta = softmax_mask(M, M_mask, axis=2)

        sum_beta = torch.sum(beta, dim=1, keepdim=True)
        docs_len = docs_len.unsqueeze(1).unsqueeze(2).expand_as(sum_beta)
        average_beta = sum_beta / docs_len.float()
        # doc-level attention
        s = torch.bmm(alpha, average_beta.transpose(1, 2))

        # predict keywords
        kws_probs = None
        if keywords is not None:
            kws_probs = []
            for i, kws in enumerate(keywords):
                document = docs_input[i].squeeze()
                cur_prob = 1.
                for j,kw in enumerate(kws):
                    if kw.data[0] == Constants.PAD:continue
                    kw = kws[j].squeeze()
                    pointer = document == kw.expand_as(document)
                    cur_prob *= torch.sum(torch.masked_select(s[i].squeeze(), pointer))
                kws_probs.append(cur_prob+1e-10)
            kws_probs = torch.cat(kws_probs, 0).squeeze()

        # predict prob of topic
        fc_feature = torch.sum(docs_outputs * s, dim=1)
        topic_probs = self.fc(fc_feature)

        return topic_probs, kws_probs, s


if __name__ == '__main__':
    from utils.Dict import Dict
    doc_input = Variable(torch.LongTensor([[1,2,3,7,12,5], [4,1,3,10,0,0]]))
    title_input = Variable(torch.LongTensor([[2, 3, 6, 0], [8, 4,  0, 0]]))

    keywords = Variable(torch.LongTensor([[2,7], [3,0]]))
    topic = Variable(torch.LongTensor([[0,1], [1, 0]]))

    vocab = Dict({i:i for i in range(40)})

    model = AoAKW(vocab_dict=vocab, dropout_rate=0.2, embed_dim=7, hidden_dim=4, n_class=2)
    for i in range(10):
        model(doc_input, title_input, keywords, topic)
