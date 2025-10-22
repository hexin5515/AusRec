'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
import random
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
import scipy.sparse as sp
import json
from tqdm import tqdm
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        #self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    print('seed:{}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    total_batch = kwargs.get('total_batch')
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, total_batch):
            yield tuple(x[i * len(x)//total_batch :(i+1) * len(x)//total_batch] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj.cpu()
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.clamp(rowsum, 0, 1000, out=None)
    d_inv_sqrt = d_inv_sqrt.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == np.inf] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    result = (adj @ d_mat_inv_sqrt).t() @ d_mat_inv_sqrt
    return result

def aux_task_load(path, dataset, aux_task_name):
    file = 'data/' + path + '/'
    A_list = {}
    with open(file + aux_task_name + '.txt', 'r') as f:
        for line in f.readlines():
            user_item = line.strip().split('\t')
            # print(user_item)
            if A_list.get(int(user_item[0])):
                A_list[int(user_item[0])].append(int(user_item[1]))
            else:
                A_list[int(user_item[0])] = []
                A_list[int(user_item[0])].append(int(user_item[1]))

    total_start = time()
    dataset: BasicDataset

    user_num = 0
    for key in A_list.keys():
        user_num += len(A_list[key])

    #user_num = user_num//20

    for i in range(dataset.n_users):
        if A_list.get(i):
            continue
        else:
            A_list[i] = []

    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = A_list
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            neguser = np.random.randint(0, dataset.n_users)
            if neguser in posForUser:
                continue
            else:
                break
        S.append([user, positem, neguser])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

def aux_task_load_(path, dataset, epoch, aux_task_name):
    file = 'data/' + path + '/'
    A_list = {}
    with open(file + aux_task_name + '.txt', 'r') as f:
        for line in f.readlines():
            user_item = line.strip().split('\t')
            # print(user_item)
            if A_list.get(int(user_item[0])):
                A_list[int(user_item[0])].append(int(user_item[1]))
            else:
                A_list[int(user_item[0])] = []
                A_list[int(user_item[0])].append(int(user_item[1]))

    total_start = time()
    dataset: BasicDataset

    user_num = 0
    for key in A_list.keys():
        user_num += len(A_list[key])

    #lastfm filmtrust
    user_num = user_num//10
    #ciao
    #user_num = user_num//20

    for i in range(dataset.n_users):
        if A_list.get(i):
            continue
        else:
            A_list[i] = []

    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = A_list
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            neguser = np.random.randint(0, dataset.n_users)
            if neguser in posForUser:
                continue
            else:
                break
        S.append([user, positem, neguser])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    #np.save('/home/miaorui/hx_project/data/' + world.dataset + '/pre/' + aux_task_name + '/' + str(epoch), np.array(S))
    return np.array(S)
    #S = np.load('/home/miaorui/hx_project/data/lastfm/pre/' + aux_task_name + '/' + str(epoch) + '.npy')
    return S

def aux_task_load_small(path, dataset, aux_task_name):
    file = 'data/' + path + '/'
    A_list = {}
    B_list = {}
    with open(file + aux_task_name + '.txt', 'r') as f:
        for line in f.readlines():
            user_item = line.strip().split('\t')
            # print(user_item)
            if A_list.get(int(user_item[0])):
                A_list[int(user_item[0])].append(int(user_item[1]))
            else:
                A_list[int(user_item[0])] = []
                A_list[int(user_item[0])].append(int(user_item[1]))

    with open(file + aux_task_name + '_all.txt', 'r') as f:
        for line in f.readlines():
            user_item = line.strip().split('\t')
            # print(user_item)
            if B_list.get(int(user_item[0])):
                B_list[int(user_item[0])].append(int(user_item[1]))
            else:
                B_list[int(user_item[0])] = []
                B_list[int(user_item[0])].append(int(user_item[1]))

    dataset: BasicDataset

    user_num = 0
    for key in A_list.keys():
        user_num += len(A_list[key])

    #user_num = user_num//20

    for i in range(dataset.n_users):
        if A_list.get(i):
            continue
        else:
            A_list[i] = []

    for i in range(dataset.n_users):
        if B_list.get(i):
            continue
        else:
            B_list[i] = []

    users = np.random.randint(0, dataset.n_users, user_num)
    Pos = A_list
    all_Pos = B_list
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = Pos[user]
        all_posForUser = all_Pos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            neguser = np.random.randint(0, dataset.n_users)
            if neguser in all_posForUser:
                continue
            else:
                break
        S.append([user, positem, neguser])
        end = time()
        sample_time1 += end - start
    return np.array(S)

def similary_base(Recmodel, pre_dataset):
    feature = Recmodel.getAllUserEmbedding().weight
    feature_t = Recmodel.getAllUserEmbedding().weight.t()

    A = feature @ feature_t
    A = normalize_adj(A)
    _, index = A.topk(10)
    A_list = {}
    for i in range(pre_dataset.n_users):
        for j in range(10):
            if A_list.get(i):
                A_list[i].append(index[i][j].item())
            else:
                A_list[i] = []
                A_list[i].append(index[i][j].item())
    user_num = 0
    for key in A_list.keys():
        user_num += len(A_list[key])
    users = np.random.randint(0, pre_dataset.n_users, user_num)
    allPos = A_list
    aux_S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            neguser = np.random.randint(0, pre_dataset.n_users)
            if neguser in posForUser:
                continue
            else:
                break
        aux_S.append([user, positem, neguser])

    return np.array(aux_S)

# ====================end Metrics=============================
# =========================================================
