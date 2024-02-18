#coding=utf-8
import torch
import numpy as np
import argparse
import torch.nn as nn
from collections import OrderedDict
from datetime import datetime
import random
import os
from tqdm import tqdm
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter


from meta_gnn.utils import set_seed
from meta_gnn.utils import load_citation_nonorm
from meta_gnn.sgc_data_generator import data_generator_ind, get_class_ind
from sklearn.preprocessing import normalize
from meta_gnn.utils_dgi import sparse_mx_to_torch_sparse_tensor

from meta_gnn.meta_dgims_3l_mi_lth import Meta


def valstep(args,j, repstep, cpt, features, pembeddings, sp_adj, class_idx, node_num, val_label, config, trainflag, device):
    meta_test_acc = []
    meta_test_loss = []
    for k in range(repstep):
        model_meta_trained = Meta(args, None, config, None, None, frozen=args.frozen, pemb=pembeddings,
                                  bottleout=args.bottleout, absin=args.debugabs, spratio=args.sparsity).to(device)
        model_meta_trained.load_state_dict(torch.load(cpt))

        if args.frozen > 0:
            if args.frozen == 1:
                for n, param in model_meta_trained.named_parameters():
                    if 'gcn' in n:
                        param.requires_grad = False
            elif args.frozen == 2:
                for n, param in model_meta_trained.named_parameters():
                    if ('gcn' in n) or ('prompt' in n):
                        param.requires_grad = False
        else:
            for n, param in model_meta_trained.named_parameters():
                assert param.requires_grad is True

        model_meta_trained.eval()
        x_spt, y_spt, x_qry, y_qry = data_generator_ind(features, class_idx, node_num, val_label, args.task_num,
                                                        args.n_way, args.k_spt, args.k_qry)
        accs, testloss, _ = model_meta_trained.forward(j, features, x_spt, y_spt, x_qry, y_qry, sp_adj,trainflag)
        meta_test_acc.append(accs)
        meta_test_loss.append(testloss)
        del model_meta_trained

    valloss = np.array(meta_test_loss).mean(axis=0)
    valacc = np.array(meta_test_acc).mean(axis=0)
    if j % 100 == 0:
        print('Step:', j, '\tMeta_val_Loss', valloss, '\tMeta_Val_Accuracy:', valacc)
    return valloss, valacc


def teststep(args, j, repstep, cpt, features, pembeddings, sp_adj, class_idx, node_num, val_label, config,trainflag, device):

    meta_test_acc = []
    meta_test_loss = []
    for k in range(repstep):
        model_meta_trained = Meta(args, None, config, None, None, frozen=args.frozen, pemb=pembeddings,
                                  bottleout=args.bottleout, absin=args.debugabs, spratio=args.sparsity).to(device)
        model_meta_trained.load_state_dict(torch.load(cpt))

        if args.frozen > 0:
            if args.frozen == 1:
                for n, param in model_meta_trained.named_parameters():
                    if 'gcn' in n:
                        param.requires_grad = False
            elif args.frozen == 2:
                for n, param in model_meta_trained.named_parameters():
                    if ('gcn' in n) or ('prompt' in n):
                        param.requires_grad = False
        else:
            for n, param in model_meta_trained.named_parameters():
                assert param.requires_grad is True

        model_meta_trained.eval()
        x_spt, y_spt, x_qry, y_qry = data_generator_ind(features, class_idx, node_num, val_label, args.task_num,
                                                        args.n_way, args.k_spt, args.k_qry)
        accs, testloss, _ = model_meta_trained.forward(j, features, x_spt, y_spt, x_qry, y_qry, sp_adj, trainflag)
        meta_test_acc.append(accs)
        meta_test_loss.append(testloss)
        del model_meta_trained

    testloss = np.array(meta_test_loss).mean(axis=0)
    testacc = np.array(meta_test_acc).mean(axis=0)
    if j % 100 == 0:
        print('Step:', j, '\tMeta_Test_Loss', testloss, '\tMeta_Test_Accuracy:', testacc)
    return testloss, testacc


def load_my_state_dict(cur, state_dict):
    own_state = cur.state_dict()
    own_state = OrderedDict(own_state)
    state_dict = OrderedDict(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)
    return own_state


def main(args):
    step = args.step
    set_seed(args.seed)
    if args.device >= 0:
        device = "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
        dataset = args.dataset
        if dataset == 'cora':
            adj, features, labels, abs_adj = load_citation_nonorm(args.dataset, args.normalization, device=device)

    else:
        device = torch.device('cpu')
        dataset = args.dataset
        if dataset == 'cora':
            adj, features, labels, abs_adj = load_citation_nonorm(args.dataset, args.normalization, device=device)

    if args.dataset == 'cora':
        node_num = 2708
        class_label = [0, 1, 2, 3, 4, 5, 6]
        np.random.shuffle(class_label)
        train_label = class_label[:3]
        val_label = class_label[3:5]
        test_label = class_label[-2:]



    abs_adj = normalize(abs_adj, norm='l1', axis=1)

    abs_adj = sparse_mx_to_torch_sparse_tensor(abs_adj).to(adj.device)

    sparse = True
    ft_size = features.shape[1]


    # use pre-trained model: dgi
    hid_units = args.hid_units
    hid_units = [int(i) for i in hid_units.split(',')]
    nonlinearity = 'prelu'

    gnn = 'dgi'
    # hid_u = hid_units
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m%d%H%M%S")
    rd = str(random.random())[2:7]
    current_save = 'CkptMs_' + args.dataset + '_N' + str(args.n_way) +'_K'+ str(args.k_spt) + '_Lr' + str(args.meta_lr)[2:] + \
                   '_Step' + str(args.update_lr)[2:] + "_"+ gnn +\
                   '_S'+ str(args.seed) + '_' + currentTime + '__C' + args.note + \
                   '_f' + str(args.frozen) + '_pr' + str(args.pregnn) + '_hid' + str(args.hid_units.replace(' ', '').replace(',', '_'))\
                   + '_Pw' + str(args.pweight) + '_Co' + str(args.enchid) + '_Sp' + str(args.sparsity) +\
                   '_abs' + str(args.debugabs)
    ckptpath = os.path.join('ckpt', current_save)
    while os.path.exists(ckptpath):

        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m%d%H%M%S")
        rd = str(random.random())[2:7]
        current_save = 'CkptMs_' + args.dataset + '_N' + str(args.n_way) +'_K'+ str(args.k_spt) + '_Lr' + str(args.meta_lr)[2:] + \
                   '_Step' + str(args.update_lr)[2:] + "_"+ gnn +\
                   '_S'+ str(args.seed) + '_' + currentTime + '__C' + args.note + \
                   '_f' + str(args.frozen) + '_pr' + str(args.pregnn) + '_hid' + str(args.hid_units.replace(' ', '').replace(',', '_')) \
                       + '_Pw' + str(args.pweight) + '_Co' + str(args.enchid)+ '_Sp' + str(args.sparsity) +\
                   '_abs' + str(args.debugabs)
        ckptpath = os.path.join('ckpt', current_save)

    os.makedirs(ckptpath)


    writer = SummaryWriter('runs/'+ current_save)
    for ep in range(1):

        encoder = DGIms(ft_size, hid_units[0], hid_units[1], args.n_way, nonlinearity, None, None, args.enchid, None,
                        args.bottleout, args.debugabs ,spratio=args.sparsity).to(device)
        if args.dgipath == '':
            dgickpt = 'best_dgi_{}_{}.pkl'.format(hid_units[0], hid_units[1])
        else:
            dgickpt = args.dgipath

        load = args.pregnn
        if load > 0:
            ckpt = torch.load(dgickpt)

            state_dict = load_my_state_dict(encoder, ckpt)
            encoder.load_state_dict(state_dict)
        else:
            print(Fore.BLUE, 'My info- No pretrain DGIms parameters loaded! ')

        config = {'name': 'dgi', 'hid': (ft_size, hid_units), 'nonlinear': nonlinearity}

        if args.frozen == 1:
            pembeddings_v = encoder.embed(features, adj, sparse = True)
            del encoder
            pf = pembeddings_v[0]
            plast = pembeddings_v[-1]
            p1 = pembeddings_v[-2]

            pf_abs = torch.spmm(abs_adj, torch.squeeze(pf, 0)).detach()
            p1_abs = torch.spmm(abs_adj, torch.squeeze(p1, 0)).detach()
            plast_abs = torch.spmm(abs_adj, torch.squeeze(plast, 0)).detach()
            pembeddings = [pf_abs, p1_abs, plast_abs,  pf, p1, plast]

            encoder = DGIms(ft_size, hid_units[0], hid_units[1], args.n_way, nonlinearity, None, None, args.enchid,
                            pembeddings, args.bottleout, args.debugabs,  spratio=args.sparsity)
            maml = Meta(args, encoder, frozen=args.frozen, bottleout=args.bottleout, absin=args.debugabs,
                        spratio=args.sparsity).to(device)

        else:

            # maml = MAML(encoder, lr=args.update_lr, first_order=not args.second_order, anil=False, allow_unused=True)
            maml = Meta(args, encoder, frozen=args.frozen, bottleout=args.bottleout, absin=args.debugabs,
                        spratio=args.sparsity).to(device)

        if args.frozen > 0:
            frozencode = args.frozen
            if frozencode == 1:
                print(Fore.BLUE, 'Frozen code ', frozencode)
                for n, param in maml.named_parameters():
                   if 'gcn' in n:
                       param.requires_grad = False

            elif frozencode == 2:
                print(Fore.BLUE, 'Frozen code ', frozencode)
                for n, param in maml.named_parameters():
                    if ('gcn' in n) or ('prompt' in n):
                        param.requires_grad = False

        else:
            assert args.frozen == 0
            for n, param in maml.named_parameters():
                assert param.requires_grad is True

        print(Fore.BLACK, 'Start training...')
        class_cls = get_class_ind(labels)

        best = np.inf
        patience = args.patience
        for j in tqdm(range(args.epoch)):

            x_spt, y_spt, x_qry, y_qry = data_generator_ind(features, class_cls, node_num, train_label, args.task_num, args.n_way, args.k_spt, args.k_qry)
            accs, trainloss, _ = maml.forward(j, features, x_spt, y_spt, x_qry, y_qry, adj, trainflag=True )
            if ep == 0:
                writer.add_scalar('loss/train0',  trainloss[0], j)
                writer.add_scalar('loss/trainl',  trainloss[-1], j)
                writer.add_scalar('acc/train0',  accs[0], j)
                writer.add_scalar('acc/trainl',  accs[-1], j)

            with open('{}/trainingloss.txt'.format(ckptpath, current_save), 'a') as f:
                if j == 0:
                    f.write('test: '+' '.join([str(lab) for lab in test_label]) + ' train:' + ' '.join([str(lab) for lab in train_label]))
                    f.write('\n')
                f.write('Cross Validation:{}, Step: {}, Meta-Train_Loss: {}'.format(ep+1, j, np.array(trainloss).astype(np.float16)))
                f.write('\n')

                # ####################### val
            if j % 100 == 0:
                torch.save(maml.state_dict(), '{}/maml_dgims_step{}_{}.pkl'.format(ckptpath, j, args.dataset))
                cpt = '{}/maml_dgims_step{}_{}.pkl'.format(ckptpath, j, args.dataset)

                valloss, valacc = valstep(args, j, args.step, cpt, features, pembeddings, adj, class_cls, node_num,
                                          val_label,
                                          config, trainflag=True, device=device)

                if args.dataset == 'cora':
                    with open('{}/cora_dgims.txt'.format(ckptpath, current_save), 'a') as f:
                        f.write('Cross Validation:{}, Step: {}, Meta-Test_Accuracy: {}'.format(ep + 1, j,
                                                                                               valacc.astype(
                                                                                                   np.float16)))
                        f.write('\n')

                if valloss[-1] < best:
                    best = valloss[-1]
                    best_t = j
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!', ep)
                    break


        # ####################### test
        if 1:

            best_tmodel = '{}/maml_dgims_step{}_{}.pkl'.format(ckptpath, best_t, args.dataset)
            repeat = 200
            meta_test_loss, meta_test_acc = teststep(args, j, repeat, best_tmodel, features, pembeddings, adj,
                                                     class_cls, node_num, test_label,
                                                     config, trainflag=True, device=device)

            if args.dataset == 'cora':
                with open('{}/cora_dgims.txt'.format(ckptpath, current_save), 'a') as f:
                    f.write('FINAL Cross Validation:{}, Step: {}, Meta-Test_Accuracy: {}'.format(best_t, j,
                                                                                                 meta_test_acc.astype(
                                                                                                     np.float16)))
                    f.write('\n')

        del encoder
        del maml


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40001)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=16)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)

    argparser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=2026, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=30, help='How many times to random select node to test')
    argparser.add_argument('--device', type=int, default=0, help='How many times to random select node to test')
    argparser.add_argument("--second_order", type=int, default=1, help="second order or not")  # 9
    argparser.add_argument('--note', type=str, default='', help='Dataset to use.')
    argparser.add_argument('--frozen', type=int, default=1, help='')
    argparser.add_argument('--pregnn', type=int, default=1, help='')
    argparser.add_argument('--hid_units', type=str, default='256, 256', help='')
    argparser.add_argument('--enchid', type=int, default=8, help='')
    argparser.add_argument('--thres', type=float, default=0.3, help='n/a')
    argparser.add_argument('--pweight', type=float, default=1.0, help='')
    argparser.add_argument('--tanhact', type=int, default=0, help='n/a')

    argparser.add_argument('--bottleout', type=int, default=24, help='n/a')
    argparser.add_argument('--dgipath', type=str, default='', help='')
    argparser.add_argument('--patience', type=int, default=60, help='')
    argparser.add_argument('--sparsity', type=float, default=0.0, help='')
    argparser.add_argument('--debugabs', type=int, default=5, help='')
    argparser.add_argument('--b2', type=int, default=1, help='')

    args = argparser.parse_args()
    if args.b2 > 0:
        from dgi.models import DGI2ms2l_mi_lth_2b as DGIms
    else:
        from dgi.models import DGI2ms2l_mi_lth as DGIms

    main(args)
