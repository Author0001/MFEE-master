# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy, time
from model.gazlstm import *
from utils.params import *

class CNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout, gpu=True):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.gpu = gpu

        self.cnn_layer0 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1) for i in range(self.num_layer-1)]
        self.drop = nn.Dropout(dropout)

        if self.gpu:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            for i in range(self.num_layer-1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()

    def forward(self, input_feature):

        batch_size = input_feature.size(0)
        seq_len = input_feature.size(1)

        input_feature = input_feature.transpose(2,1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)  #(b,h,l)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer-1):
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2,1).contiguous()
        return cnn_output



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  ## (b,h,l,d) * (b,h,d,l)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   ##(b,h,l,l) * (b,h,l,d) = (b,h,l,d)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class AttentionModel(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        c = copy.deepcopy
        # attn0 = MultiHeadedAttention(head, d_input, d_model)
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = PositionalEncoding(d_model, dropout)
        # layer0 = EncoderLayer(d_model, c(attn0), c(ff), dropout)
        layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.layers = clones(layer, num_layer)
        # layerlist = [copy.deepcopy(layer0),]
        # for _ in range(num_layer-1):
        #     layerlist.append(copy.deepcopy(layer))
        # self.layers = nn.ModuleList(layerlist)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input, d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # x: embedding (b,l,we)
        x = self.posi(self.input2model(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)




class NERmodel(nn.Module):

    def __init__(self, model_type, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True, biflag=True):
        super(NERmodel, self).__init__()
        self.model_type = model_type

        if self.model_type == 'lstm':
            config = parse_args()
            self.type_cls = TypeCls(config)
            #self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
            #self.drop = nn.Dropout(dropout)

        if self.model_type == 'cnn':
            self.cnn = CNNmodel(input_dim, hidden_dim, num_layer, dropout, gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.attention_model = AttentionModel(d_input=input_dim, d_model=hidden_dim, d_ff=2*hidden_dim, head=4, num_layer=num_layer, dropout=dropout)
            for p in self.attention_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, input, mask=None):

        if self.model_type == 'lstm':
            feature_out_d = self.type_cls(input, mask)
            #hidden = None            
            #feature_out, hidden = self.lstm(input, hidden)
            #feature_out_d = self.drop(feature_out)

        if self.model_type == 'cnn':
            feature_out_d = self.cnn(input)

        if self.model_type == 'transformer':
            feature_out_d = self.attention_model(input, mask)

        return feature_out_d


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################



from torch import nn
import math
import torch


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    def forward(self, x, condition):
        '''

        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        condition = condition.unsqueeze(1).expand_as(x)
        gamma = self.gamma_dense(condition) + self.gamma
        beta = self.beta_dense(condition) + self.beta
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x


class AdaptiveAdditionPredictor(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(AdaptiveAdditionPredictor, self).__init__()
        self.v = nn.Linear(hidden_size * 4, 1)
        self.hidden = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, context, mask):
        '''
        :param query: [c, e]
        :param context: [b, t, e]
        :param mask: [b, t], 0 if masked
        :return: [b, e]
        '''

        context_ = context.unsqueeze(1).expand(context.size(0), query.size(0), context.size(1), context.size(2))  # [b, c, t, e]
        query_ = query.unsqueeze(0).unsqueeze(2).expand_as(context_)  # [b, c, t, e]

        scores = self.v(torch.tanh(self.hidden(torch.cat([query_, context_, torch.abs(query_ - context_), query_ * context_], dim=-1))))  # [b, c, t, 1]
        scores = self.dropout(scores)
        mask = (mask < 1).unsqueeze(1).unsqueeze(3).expand_as(scores)  # [b, c, t, 1]
        scores = scores.masked_fill_(mask, -1e10)
        scores = scores.transpose(-1, -2)  # [b, c, 1, t]
        scores = torch.softmax(scores, dim=-1)  # [b, c, 1, t]
        g = torch.matmul(scores, context_).squeeze(2)  # [b, c, e]
        query = query.unsqueeze(0).expand_as(g)  # [b, c, e]

        pred = self.v(torch.tanh(self.hidden(torch.cat([query, g, torch.abs(query - g), query * g], dim=-1)))).squeeze(-1)  # [b, c]
        return pred


class MultiHeadedAttention1(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention1, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask. \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output
