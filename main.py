import os
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
import matplotlib.pyplot as plt
from model import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorboardX import SummaryWriter
from math import sqrt
import world
import utils
import time
from utils import timer
from os.path import join
from world import cprint
import Procedure
import register
from register import pre_dataset
import os
def train(Recmodel, vnet, phis,
              device, pre_dataset, S, aux_task1, aux_task2, aux_task3, aux_task4, aux_task5, aux_task6, aux_task7,
              optimizer, optimizer_vnet, opt_phi,
              epoch, loss_class):
    prop = 0.7

    pre_weight = 0
    aux1_weight = 0
    aux2_weight = 0
    aux3_weight = 0
    aux4_weight = 0
    aux5_weight = 0
    aux6_weight = 0
    aux7_weight = 0


    Recmodel.train()

    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()


    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)


    aux1_users = torch.Tensor(aux_task1[:, 0]).long().to(world.device)
    aux1_posUsers = torch.Tensor(aux_task1[:, 1]).long().to(world.device)
    aux1_negUsers = torch.Tensor(aux_task1[:, 2]).long().to(world.device)

    aux2_users = torch.Tensor(aux_task2[:, 0]).long().to(world.device)
    aux2_posUsers = torch.Tensor(aux_task2[:, 1]).long().to(world.device)
    aux2_negUsers = torch.Tensor(aux_task2[:, 2]).long().to(world.device)

    aux3_users = torch.Tensor(aux_task3[:, 0]).long().to(world.device)
    aux3_posUsers = torch.Tensor(aux_task3[:, 1]).long().to(world.device)
    aux3_negUsers = torch.Tensor(aux_task3[:, 2]).long().to(world.device)

    aux4_users = torch.Tensor(aux_task4[:, 0]).long().to(world.device)
    aux4_posUsers = torch.Tensor(aux_task4[:, 1]).long().to(world.device)
    aux4_negUsers = torch.Tensor(aux_task4[:, 2]).long().to(world.device)

    aux5_users = torch.Tensor(aux_task5[:, 0]).long().to(world.device)
    aux5_posUsers = torch.Tensor(aux_task5[:, 1]).long().to(world.device)
    aux5_negUsers = torch.Tensor(aux_task5[:, 2]).long().to(world.device)

    aux6_users = torch.Tensor(aux_task6[:, 0]).long().to(world.device)
    aux6_posUsers = torch.Tensor(aux_task6[:, 1]).long().to(world.device)
    aux6_negUsers = torch.Tensor(aux_task6[:, 2]).long().to(world.device)

    aux7_users = torch.Tensor(aux_task7[:, 0]).long().to(world.device)
    aux7_posUsers = torch.Tensor(aux_task7[:, 1]).long().to(world.device)
    aux7_negUsers = torch.Tensor(aux_task7[:, 2]).long().to(world.device)


    total_batch = len(users) // world.config['bpr_batch_size'] + 1


    Loss = 0.

    pre_x_input = torch.tensor([])

    for (batch_i,
            (batch_users,
             batch_pos,
             batch_neg,
             batch_aux1_users,
             batch_aux1_posUsers,
             batch_aux1_negUsers,
             batch_aux2_users,
             batch_aux2_posUsers,
             batch_aux2_negUsers,
             batch_aux3_users,
             batch_aux3_posUsers,
             batch_aux3_negUsers,
             batch_aux4_users,
             batch_aux4_posUsers,
             batch_aux4_negUsers,
             batch_aux5_users,
             batch_aux5_posUsers,
             batch_aux5_negUsers,
             batch_aux6_users,
             batch_aux6_posUsers,
             batch_aux6_negUsers,
             batch_aux7_users,
             batch_aux7_posUsers,
             batch_aux7_negUsers,
          )) in enumerate(utils.minibatch(users,
                                          posItems,
                                          negItems,
                                          aux1_users,
                                          aux1_posUsers,
                                          aux1_negUsers,
                                          aux2_users,
                                          aux2_posUsers,
                                          aux2_negUsers,
                                          aux3_users,
                                          aux3_posUsers,
                                          aux3_negUsers,
                                          aux4_users,
                                          aux4_posUsers,
                                          aux4_negUsers,
                                          aux5_users,
                                          aux5_posUsers,
                                          aux5_negUsers,
                                          aux6_users,
                                          aux6_posUsers,
                                          aux6_negUsers,
                                          aux7_users,
                                          aux7_posUsers,
                                          aux7_negUsers,
                                          total_batch=total_batch,
                                          batch_size=world.config['bpr_batch_size'])):

        negUsers_list_aux = []
        negUsers_list_aux.append(batch_aux1_negUsers)
        negUsers_list_aux.append(batch_aux2_negUsers)
        negUsers_list_aux.append(batch_aux3_negUsers)
        negUsers_list_aux.append(batch_aux4_negUsers)
        negUsers_list_aux.append(batch_aux5_negUsers)
        negUsers_list_aux.append(batch_aux6_negUsers)
        negUsers_list_aux.append(batch_aux7_negUsers)

        users_list_aux = []
        users_list_aux.append(batch_aux1_users)
        users_list_aux.append(batch_aux2_users)
        users_list_aux.append(batch_aux3_users)
        users_list_aux.append(batch_aux4_users)
        users_list_aux.append(batch_aux5_users)
        users_list_aux.append(batch_aux6_users)
        users_list_aux.append(batch_aux7_users)

        posUsers_list_aux = []
        posUsers_list_aux.append(batch_aux1_posUsers)
        posUsers_list_aux.append(batch_aux2_posUsers)
        posUsers_list_aux.append(batch_aux3_posUsers)
        posUsers_list_aux.append(batch_aux4_posUsers)
        posUsers_list_aux.append(batch_aux5_posUsers)
        posUsers_list_aux.append(batch_aux6_posUsers)
        posUsers_list_aux.append(batch_aux7_posUsers)


        l_g_meta = 0
        meta_Recmodel = register.MODELS[world.model_name](world.config, pre_dataset)
        meta_Recmodel = meta_Recmodel.to(world.device)
        meta_Recmodel.load_state_dict(Recmodel.state_dict())

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = meta_Recmodel.getEmbedding(batch_users[:int(len(batch_users) * prop)], batch_pos[:int(len(batch_users) * prop)], batch_neg[:int(len(batch_users) * prop)])

        reg_loss = (1 / 2) * (userEmb0.norm(2, dim=1).pow(2) +
                              posEmb0.norm(2, dim=1).pow(2) +
                              negEmb0.norm(2, dim=1).pow(2)) / float(int(len(batch_users) * prop))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.nn.functional.softplus(neg_scores - pos_scores)

        reg_loss = reg_loss * world.config['decay']
        pre_cost = loss + reg_loss

        cost_v = torch.reshape(pre_cost, (len(pre_cost), 1))

        type_v = torch.stack([torch.zeros(7)] * len(cost_v), dim=0).to(world.device)

        meta_type, meta_cost, meta_cost_input = [], [], []

        optimizer_vnet.zero_grad()

        for i in range(7):
            meta_type_aux = torch.eye(7)[i]

            (meta_users_emb, meta_pos_emb, meta_neg_emb,
             meta_userEmb0, meta_posEmb0, meta_negEmb0) = meta_Recmodel.getUserEmbedding(users_list_aux[i],
                                                                                         posUsers_list_aux[i],
                                                                                         negUsers_list_aux[i])

            meta_reg_loss = (1 / 2) * (meta_userEmb0.norm(2, dim=1).pow(2) +
                                       meta_posEmb0.norm(2, dim=1).pow(2) +
                                       meta_negEmb0.norm(2, dim=1).pow(2)) / float(len(batch_aux1_users))

            meta_pos_scores = torch.mul(meta_users_emb, meta_pos_emb)
            meta_pos_scores = torch.sum(meta_pos_scores, dim=1)
            meta_neg_scores = torch.mul(meta_users_emb, meta_neg_emb)
            meta_neg_scores = torch.sum(meta_neg_scores, dim=1)

            meta_loss = torch.nn.functional.softplus(meta_neg_scores - meta_pos_scores)

            meta_reg_loss = meta_reg_loss * world.config['decay']
            m_cost = meta_loss + meta_reg_loss
            m_cost = torch.reshape(m_cost, (len(m_cost), 1))
            meta_cost.append(m_cost)
            cost_t = torch.stack([meta_type_aux] * len(m_cost), dim=0).to(device)
            meta_type.append(cost_t)

        type_m = torch.cat((meta_type)).to(world.device)
        cost_m = torch.cat((meta_cost))
        types = torch.cat((type_v, type_m))
        costs = torch.cat((cost_v, cost_m))
        inputs = torch.cat((costs, types), 1)

        v_lambdaa = vnet(inputs.data)
        v_lambda = v_lambdaa.clone()
        v_lambda[int(len(batch_users) * prop):] = v_lambda[int(len(batch_users) * prop):] / 7

        l_f_meta = (torch.sum(cost_v) + (torch.sum(costs[int(len(batch_users) * prop):] * v_lambda[int(len(batch_users) * prop):])))/len(costs)

        meta_Recmodel.update_params(lr_inner=0.001, source_params=torch.autograd.grad(l_f_meta, (meta_Recmodel.params()), create_graph=True))

        (v_users_emb, v_pos_emb, v_neg_emb,
         v_userEmb0, v_posEmb0, v_negEmb0) = meta_Recmodel.getEmbedding(batch_users[int(len(batch_users) * prop):], batch_pos[int(len(batch_users) * prop):], batch_neg[int(len(batch_users) * prop):])

        v_reg_loss = (1 / 2) * (v_userEmb0.norm(2).pow(2) +
                              v_posEmb0.norm(2).pow(2) +
                              v_negEmb0.norm(2).pow(2)) / float(int(len(batch_users) * (1 - prop)))

        v_pos_scores = torch.mul(v_users_emb, v_pos_emb)
        v_pos_scores = torch.sum(v_pos_scores, dim=1)
        v_neg_scores = torch.mul(v_users_emb, v_neg_emb)
        v_neg_scores = torch.sum(v_neg_scores, dim=1)

        v_loss = torch.mean(torch.nn.functional.softplus(v_neg_scores - v_pos_scores))

        v_reg_loss = v_reg_loss * world.config['decay']
        v_pre_cost = v_loss + v_reg_loss

        l_g_meta += v_pre_cost

        optimizer_vnet.zero_grad()

        l_g_meta.backward()

        optimizer_vnet.step()

        (out_users_emb, out_pos_emb, out_neg_emb,
         out_userEmb0, out_posEmb0, out_negEmb0) = Recmodel.getEmbedding(batch_users, batch_pos, batch_neg)

        out_reg_loss = (1 / 2) * (out_userEmb0.norm(2, dim=1).pow(2) +
                                out_posEmb0.norm(2, dim=1).pow(2) +
                                out_negEmb0.norm(2, dim=1).pow(2)) / float(len(batch_users))


        out_pos_scores = torch.mul(out_users_emb, out_pos_emb)
        out_pos_scores = torch.sum(out_pos_scores, dim=1)
        out_neg_scores = torch.mul(out_users_emb, out_neg_emb)
        out_neg_scores = torch.sum(out_neg_scores, dim=1)

        out_loss = torch.nn.functional.softplus(out_neg_scores - out_pos_scores)

        out_reg_loss = out_reg_loss * world.config['decay']
        out_pre_cost = out_loss + out_reg_loss

        optimizer.zero_grad()

        out_cost_v = torch.reshape(out_pre_cost, (len(out_pre_cost), 1))
        out_type_v = torch.stack([torch.zeros(7)] * len(out_cost_v), dim=0)

        out_type, out_cost, out_cost_input = [], [], []

        for j in range(7):
            meta_type = torch.eye(7)[j]

            (out_users_emb_aux, out_pos_emb_aux, out_neg_emb_aux,
             out_userEmb0_aux, out_posEmb0_aux, out_negEmb0_aux) = Recmodel.getUserEmbedding(users_list_aux[j],
                                                                                             posUsers_list_aux[j],
                                                                                             negUsers_list_aux[j])

            out_reg_loss = (1 / 2) * (out_userEmb0_aux.norm(2, dim=1).pow(2) +
                                      out_posEmb0_aux.norm(2, dim=1).pow(2) +
                                      out_negEmb0_aux.norm(2, dim=1).pow(2)) / float(len(batch_aux1_users))

            out_pos_scores = torch.mul(out_users_emb_aux, out_pos_emb_aux)
            out_pos_scores = torch.sum(out_pos_scores, dim=1)
            out_neg_scores = torch.mul(out_users_emb_aux, out_neg_emb_aux)
            out_neg_scores = torch.sum(out_neg_scores, dim=1)

            out_loss = torch.nn.functional.softplus(out_neg_scores - out_pos_scores)

            out_reg_loss = out_reg_loss * world.config['decay']
            out_aux_cost = out_loss + out_reg_loss

            out_aux_cost = torch.reshape(out_aux_cost, (len(out_aux_cost), 1))
            out_cost.append(out_aux_cost)

            cost_t = torch.stack([meta_type] * len(out_aux_cost), dim=0)
            out_type.append(cost_t)

        out_type_m = torch.cat((out_type)).to(world.device)
        out_cost_m = torch.cat((out_cost))
        out_types = torch.cat((out_type_v.to(world.device), out_type_m.to(world.device)))
        out_costs = torch.cat((out_cost_v.to(world.device), out_cost_m.to(world.device)))
        out_inputs = torch.cat(((out_costs).to(world.device), out_types.to(world.device)), 1)

        with torch.no_grad():
            w_new = vnet(out_inputs.data)



        w_new[len(batch_users):] = w_new[len(batch_users):]/7
        loss = (torch.sum(out_cost_v) + torch.sum(out_costs[len(batch_users):] * w_new[len(batch_users):]))/len(out_costs)

        loss.backward()
        optimizer.step()

        w_new[len(batch_users):] = (w_new[len(batch_users):] - min(w_new[len(batch_users):])) / (max(w_new[len(batch_users):]) - min(w_new[len(batch_users):]))

        pre_weight += sum(w_new[:len(batch_users)]).item()
        aux1_weight += sum(w_new[len(batch_users):len(batch_users) + len(batch_aux1_users)]).item()
        aux2_weight += sum(w_new[len(batch_users) + len(batch_aux1_users):len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users)]).item()
        aux3_weight += sum(w_new[len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users):len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users)]).item()
        aux4_weight += sum(w_new[len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users):
                                 len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users)]).item()
        aux5_weight += sum(w_new[len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users):
                                 len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users) + len(batch_aux5_users)]).item()
        aux6_weight += sum(w_new[len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users) + len(batch_aux5_users):
                                 len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users) + len(batch_aux5_users) + len(batch_aux6_users)]).item()
        aux7_weight += sum(w_new[len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users) + len(batch_aux5_users) + len(batch_aux6_users):
                                 len(batch_users) + len(batch_aux1_users) + len(batch_aux2_users) + len(batch_aux3_users) + len(batch_aux4_users) + len(batch_aux5_users) + len(batch_aux6_users) + len(batch_aux7_users)]).item()

        Loss += (torch.sum(out_cost_v)/len(out_cost_v))/total_batch


    pre_y = pre_weight / len(users)
    aux1_y = aux1_weight/len(aux1_users)
    aux2_y = aux2_weight/len(aux2_users)
    aux3_y = aux3_weight/len(aux3_users)
    aux4_y = aux4_weight / len(aux4_users)
    aux5_y = aux5_weight / len(aux5_users)
    aux6_y = aux6_weight / len(aux6_users)
    aux7_y = aux7_weight / len(aux7_users)

    return pre_y, aux1_y, aux2_y, aux3_y, aux4_y, aux5_y, aux6_y, aux7_y, Loss.item()#pre_y, aux1_y, aux2_y, aux3_y, aux4_y, aux5_y, aux6_y, aux7_y, aux8_y,


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def reverse_dict(d):
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[str(v)].append(int(k))
    return dict(re_d)


def main():
    utils.set_seed(world.seed)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    Recmodel = register.MODELS[world.model_name](world.config, pre_dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")


    print(Recmodel.params())
    optimizer = torch.optim.Adam(Recmodel.params(), lr=world.config['lr'])


    ### Task-specific layers ###
    phis, opt_phi = [], []
    for _ in range(7):
        phi = Phi(world.config['latent_dim_rec'], world.config['latent_dim_rec']).to(world.device)
        opt_phi.append(torch.optim.Adam(phi.params(), lr=world.config['lr']))
        phis.append(phi)

    ### Weighting function ###
    input_dim = 8
    vnet = Weight(input_dim, 1000, 1).to(world.device)
    optimizer_vnet = torch.optim.Adam(vnet.params(), lr=world.config['wlr'])



    best_recall = 0
    best_ndcg = 0
    best_ndcg_epoch = 0
    best_recall_epoch = 0


    x_input = range(world.TRAIN_epochs + 1)
    #x_input = range(2)
    pre_loss = []
    pre_y = []
    aux1_y = []
    aux2_y = []
    aux3_y = []
    aux4_y = []
    aux5_y = []
    aux6_y = []
    aux7_y = []



    for epoch in range(world.TRAIN_epochs + 1):

        # lastfm dbook
        S = utils.UniformSample_original(pre_dataset)
        aux_task1 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'H_s')
        aux_task2 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'H_j')
        aux_task3 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'trustnetwork')
        aux_task4 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'two_hop')
        aux_task5 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'three_hop')
        aux_task6 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'uiu')
        aux_task7 = utils.aux_task_load_(world.dataset, pre_dataset, epoch, 'uuiu')

        if epoch % 10 == 0:
            cprint("[TEST]")
            result = Procedure.Test(pre_dataset, Recmodel, epoch, w, world.config['multicore'])
            #os.system("echo " + "epoch:{}".format(epoch) + "ndcg={}".format(result['ndcg']) + "recall = {}".format(result['recall']) + ">> result.txt")
        pre_task_w, aux1_task_w, aux2_task_w, aux3_task_w, aux4_task_w,\
        aux5_task_w, aux6_task_w, aux7_task_w, Loss = train(Recmodel, vnet, phis, #aux8_task_w,
              world.device, pre_dataset, S, aux_task1, aux_task2, aux_task3, aux_task4, aux_task5, aux_task6, aux_task7, #aux_task8,
              optimizer, optimizer_vnet, opt_phi,
              epoch, bpr,)

        pre_y.append(pre_task_w)
        aux1_y.append(aux1_task_w)
        aux2_y.append(aux2_task_w)
        aux3_y.append(aux3_task_w)
        aux4_y.append(aux4_task_w)
        aux5_y.append(aux5_task_w)
        aux6_y.append(aux6_task_w)
        aux7_y.append(aux7_task_w)
        pre_loss.append(Loss)

        print("epoch={},loss:{}".format(epoch, round(Loss, 3)))

if __name__ == "__main__":
    main()


'''
json_str = json.dumps(m_users)
print(json_str)
with open('./data/movielens/m_users.json', 'w') as json_file:
    json_file.write(json_str)
'''


