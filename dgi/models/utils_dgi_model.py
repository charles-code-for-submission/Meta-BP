import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from meta_gnn.normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
from meta_gnn.utils import load_citation
import torch.nn as nn
# adj, features, labels = load_citation('cora', 'AugNormAdj', cuda=False)

# print(adj)
import torch.nn as nn
import torch.nn.functional as F #.dropout(input, p=0.5, training=True, inplace=False)


import math

def get_neighbors(features, adj, thres):
    source = adj._indices()[0]
    target = adj._indices()[1]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    fs = features[source]
    ft = features[target]
    scores = cos(fs, ft)

    lc = scores >= thres
    mc = scores < thres
    lscores = scores[lc]
    mscores = scores[mc]

    # get indices
    indices = adj._indices().T

    lindices = indices[lc, :].T
    mindices = indices[mc, :].T
    nodenum = features.shape[0]
    ladj = torch.sparse_coo_tensor(lindices, lscores, (nodenum, nodenum))
    madj = torch.sparse_coo_tensor(mindices, mscores, (nodenum, nodenum))

    return ladj, madj


def funcfenc(hin, hout, nonl, bias, hid, return_sp=False):
    net = nn.Sequential(nn.Linear(hin, hid,  bias),  nn.Linear(hid, hout,  bias))
    return net


def funcf(hin, hout, nonl, bias):
    net = nn.Linear(hin, hout,  bias)
    return net


def funcg(hin, hout, nonl, bias):
    net = nn.Linear(hin, hout, bias)
    return net


def aggrm(h, hm1, ladj, madj, f, g, sparse=True):

    mf = f(h)
    mg = g(h)
    if sparse:
        aggf = torch.spmm(ladj, torch.squeeze(hm1, 0))
    else:
        aggf = torch.bmm(ladj, hm1)

    if sparse:
        aggg = torch.spmm(madj, torch.squeeze(hm1, 0))
    else:
        aggg = torch.bmm(madj, hm1)

    return mf, mg, aggf, aggg


def aggrmwPureS(h, hm1, ladj, madj, wf, wf2, wg, sparse, sc1, sc2, ratio, mode, keepmask=None, printS=False):
    sparse = True
    assert ladj is None
    w_pruned, b_pruned = None, None
    if mode == "train":
        weight_masks = []
        weight_mask1 = GetSubnetFaster.apply(sc1.abs(),
                                            torch.zeros(sc1.shape),
                                            torch.ones(sc1.shape),
                                            ratio)

        weight_mask2 = GetSubnetFaster.apply(sc2.abs(),
                                        torch.zeros(sc2.shape),
                                        torch.ones(sc2.shape),
                                        ratio)
        pruned_wf = wf.T * weight_mask1

        pruned_wf2 = wf2.T * weight_mask2
        hd = h

        mf1 = torch.matmul(hd, pruned_wf)

        mf2 = torch.matmul(mf1, pruned_wf2)

        if printS:
            print('weight1', weight_mask1)
            print('weight2', weight_mask2)
            print('sp', ratio)

        return mf2, weight_mask1, weight_mask2, ratio

    elif mode == "valid":
        keepmask1, keepmask2 = keepmask

        pruned_wf = wf.T * keepmask1

        pruned_wf2 = wf2.T * keepmask2

        hd = h

        mf = torch.matmul(hd, pruned_wf)

        mf = torch.matmul(mf, pruned_wf2)
        return mf, None, None, None

    elif mode == "test":
        keepmask1, keepmask2 = keepmask

        pruned_wf = wf.T * keepmask1

        pruned_wf2 = wf2.T * keepmask2

        hd = h

        mf = torch.matmul(hd, pruned_wf)

        mf = torch.matmul(mf, pruned_wf2)
        return mf, None, None, None


def aggrmw(h, hm1, ladj, madj, wf,wf2, wg, sparse=True):

    if ladj is None:
        hd = h
        hm1d = hm1
        mf = torch.matmul(hd, wf.T)

        mf = torch.matmul(mf, wf2.T)

        mg = torch.matmul(hd, wg.T)

        return mf, mg, None, None

    else:
        hd = h
        hm1d = hm1
        mf = torch.matmul(hd, wf.T)

        mf = torch.matmul(mf, wf2.T)

        mg = torch.matmul(hd, wg.T)

        if sparse:
            aggf = torch.spmm(ladj, torch.squeeze(hm1d, 0))
        else:
            aggf = torch.bmm(ladj, hm1d)

        if sparse:
            aggg = torch.spmm(madj, torch.squeeze(hm1d, 0))
        else:
            aggg = torch.bmm(madj, hm1d)
        return mf, mg, aggf, aggg


# todo discarded loss
def lossm(mf, aggf):
    loss = nn.L1Loss(reduction='none')
    l = torch.sum(loss(mf, aggf), dim=-1)/16
    l = torch.mean(l)
    return l


def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()


class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity*100)
        # print('foraward val', k_val)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def aggrmwPure3l(h, hm1, f0,  ladj, madj, wf0, wf0b, wf02, wf02b,  wfin, wfinb, wfin2,  wfin2b, wf, wfb, wf2,wf2b,
                 wg, sparse=True, bias=False, act=False, train=True):
    assert ladj is None

    if ladj is None:
        hf0 = f0
        hf0_mid = torch.matmul(hf0, wf0.T)
        if bias:
            hf0_mid = hf0_mid + wf0b
        if act:
            hf0_mid = torch.relu(hf0_mid)

        hf0_out = torch.matmul(hf0_mid, wf02.T)
        if bias:
            hf0_out = hf0_out + wf02b
        if act:
            hf0_out = torch.relu(hf0_out)

        hin = hm1
        hin_mid = torch.matmul(hin, wfin.T)
        if bias:
            hin_mid = hin_mid + wfinb
        if act:
            hin_mid = torch.relu(hin_mid)

        hin_out = torch.matmul(hin_mid, wfin2.T)
        if bias:
            hin_out = hin_out + wfin2b
        if act:
            hin_out = torch.relu(hin_out)

        hd = h
        hdmid = torch.matmul(hd, wf.T)
        if bias:
            hdmid = hdmid + wfb
        if act:
            hdmid = torch.relu(hdmid)

        hdout = torch.matmul(hdmid, wf2.T)
        if bias:
            hdout = hdout + wf2b
        if act:
            hdout = torch.relu(hdout)

        return hdout, hin_out, hf0_out,  hdmid, hin_mid, hf0_mid


def aggrmwPure3lSP(h, hm1, f0,  ladj, madj, wf0, wf0b, wf02, wf02b,  wfin, wfinb, wfin2,  wfin2b, wf, wfb, wf2,wf2b, wg,
                   sparse=True, bias=False, act=False, train=True, sc1=None, sc2=None, ratio=0, keepmask=None):
    assert ladj is None
    #
    if ratio > 0:
        if train is True:
            weight_mask1 = GetSubnetFaster.apply(sc1.abs(),
                                                torch.zeros(sc1.shape),
                                                torch.ones(sc1.shape),
                                                ratio)

            weight_mask2 = GetSubnetFaster.apply(sc2.abs(),
                                            torch.zeros(sc2.shape),
                                            torch.ones(sc2.shape),
                                            ratio)

            pruned_wfin = wfin.T * weight_mask1
            pruned_wf = wf.T * weight_mask2

            hin = hm1
            hin_mid = torch.matmul(hin, pruned_wfin)
            if bias:
                hin_mid = hin_mid + wfinb
            if act:
                hin_mid = torch.relu(hin_mid)

            hd = h
            hdmid = torch.matmul(hd, pruned_wf)
            if bias:
                hdmid = hdmid + wfb
            if act:
                hdmid = torch.relu(hdmid)
            return None, None, None,  hdmid, hin_mid, None

        else:
            keepmask1, keepmask2 = keepmask
            pruned_wfin = wfin.T * keepmask1
            pruned_wf = wf.T * keepmask2

            hin = hm1
            hin_mid = torch.matmul(hin, pruned_wfin.T)
            if bias:
                hin_mid = hin_mid + wfinb
            if act:
                hin_mid = torch.relu(hin_mid)

            hd = h
            hdmid = torch.matmul(hd, pruned_wf.T)
            if bias:
                hdmid = hdmid + wfb
            if act:
                hdmid = torch.relu(hdmid)

            return None, None, None,  hdmid, hin_mid, None

    else:
        assert ratio == 0

        if ladj is None:

            hin = hm1
            hin_mid = torch.matmul(hin, wfin.T)

            if bias:
                hin_mid = hin_mid + wfinb
            if act:
                hin_mid = torch.relu(hin_mid)

            hd = h

            hdmid = torch.matmul(hd, wf.T)
            if bias:
                hdmid = hdmid + wfb
            if act:
                hdmid = torch.relu(hdmid)


            return None, None, None,  hdmid, hin_mid, None