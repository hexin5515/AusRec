import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
import argparse
from tqdm import tqdm
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dataloader
import world
from dataloader import BasicDataset

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        #print("-----------------")
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if hasattr(curr_module, 'named_leaves'):
            #print(curr_module.named_leaves())
            for name, p in curr_module.named_leaves():
                #print("{}.size():{}".format(name, p.size()))
                if p is not None and p not in memo:
                    memo.add(p)
                    #print(prefix + ('.' if prefix else '') + name)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    #print(prefix + ('.' if prefix else '') + name)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                #print("{}.size():{}".format(name, p.size()))
                yield name, p


    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):

        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                # print(name_t)
                # print(param_t.size())
                # print(grad.size())
                # print("\n")
                tmp = param_t - lr_inner * grad
                # if "vnet" in kwargs:
                #     xxxx = 1
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            #print("更新参数成功！\n")

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBN(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(192)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.batch_norm(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaEmb(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Embedding(*args, **kwargs)
        nn.init.xavier_normal_(ignore.weight)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

    def forward(self, x):
        return F.embedding(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class LightGCN(MetaModule):

    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = MetaEmb(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = MetaEmb(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            #world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        #print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # print(all_emb)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                #print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # print(g_droped)
                # a = []
                # print(a[10])
                # print(g_droped.size())
                # print(all_emb.size())
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # print(users)
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        return users_emb, items_emb

    def getEmbedding(self, users, pos_items, neg_items = None):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getUserEmbedding(self, users, pos_users, neg_users = None):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_users[pos_users]
        neg_emb = all_users[neg_users]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_user(pos_users)
        neg_emb_ego = self.embedding_user(neg_users)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getItemEmbedding(self):
        return self.embedding_item

    def getAllUserEmbedding(self):
        return self.embedding_user


    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        # inner_pro = torch.mul(users_emb, items_emb)
        # gamma = torch.sum(inner_pro, dim=1)
        return users_emb,items_emb

class Weight(MetaModule):
    def __init__(self, in_channel, hidden_1, out_channel):#hidden_2,
        super(Weight, self).__init__()
        self.linear1 = MetaLinear(in_channel, hidden_1)
        self.linear2 = MetaLinear(hidden_1, out_channel)#out_channel
        #self.linear3 = MetaLinear(hidden_2, out_channel)
        self.BN = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = self.linear1(x)
        x = self.BN(x)
        x = torch.relu(x)
        x = self.linear2(x)
        # x = torch.relu(x)
        # x = self.linear3(x)
        x = torch.sigmoid(x)
        return x


class Phi(MetaModule):
    #def __init__(self, in_channel, hidden, out_channel):
    def __init__(self, in_channel, out_channel):
        super(Phi, self).__init__()

        self.linear1 = MetaLinear(in_channel, out_channel)
        #self.linear2 = MetaLinear(hidden, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        # x = F.relu(x)
        # x = self.linear2(x)
        return x