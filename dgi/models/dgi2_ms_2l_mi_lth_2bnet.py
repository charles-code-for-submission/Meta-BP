import torch
import torch.nn as nn
from dgi.layers import GCN, AvgReadout, Discriminator
import torch.nn.functional as F
#from dgi.models.utils_prompt import get_neighbors, aggrm, aggrmw, funcf, funcg, lossm, lossml2
from dgi.models.utils_promptms_2l import funcfenc, aggrmw2l, get_neighbors, lossm, lossml2, funcg, aggrmwPure2l, \
    aggrmwPure3l, aggrmwPure3lSP
import numpy as np
import math

def constr( bn1, bn2):
    nodenumber = bn1.shape[0]

    def dotproduct(x, y):
        xnorm = F.normalize(x, dim=-1)
        ynorm = F.normalize(y, dim=-1)
        dotp = xnorm * ynorm
        return dotp.sum(-1)

    def samplePN(nodenumber):
        # posn = nodenumber
        all_list = [(i, j) for i in range(nodenumber) for j in range(nodenumber)]
        pos_list = [(i, i) for i in range(nodenumber)]

        neg_list_all = [pair for pair in all_list if pair not in pos_list]

        neglen = len(neg_list_all)
        # print(neg_list_all, neglen)
        neg_ind = np.random.choice(range(neglen), 2 * nodenumber, replace=False)

        pos_ar = np.array(pos_list)
        neg_ar_all = np.array(neg_list_all)

        neg_ar = neg_ar_all[neg_ind]
        return pos_ar, neg_ar

    ppair, npair = samplePN(nodenumber)

    # plabel = torch.ones((ppair.shape, 1))
    # nlabel = torch.ones((npair.shape, 0))

    possim = dotproduct(bn1[ppair[:, 0]], bn2[ppair[:, 1]])
    # print(possim, possim.shape)
    # posdistance = torch.ones(possim.shape) - possim
    # print(npair, ppair, npair.shape, ppair.shape)
    negsim = dotproduct(bn1[npair[:, 0]], bn2[npair[:, 1]])
    # print(negsim, negsim.shape)
    # negdistance = torch.ones(negsim.shape) - negsim
    distance = max(0, negsim.mean() - possim.mean())
    # print(distance)
    return distance


from meta_gnn.mine.models.mine import get_est
class DGI2ms2l_mi_lth_2b(nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_way, activation, ladj, madj, hid, pemb=None, bottleout=None, absin=0, spratio=0.0):
        super(DGI2ms2l_mi_lth_2b, self).__init__()
        self.gcn1 = GCN(n_in,  n_h1, activation)
        self.gcn2 = GCN(n_h1, n_h2, activation)
        # print('load 2b')
        self.vars = nn.ParameterList()
        self.read = AvgReadout()

        self.outdim = n_h2
        self.n_way = n_way
        # todo 已经去除sigmoid
        self.layer = 2
        self.hid = hid
        self.bottleout = bottleout
        # self.bottleout = self.hid
        self.bottleout = 1

        assert n_h2 == n_h1

        self.prompt_mi1 = get_est(n_h1,
                                self.hid)
        self.prompt_mi2 = get_est(n_h2,
                                  self.hid)
        # only x and  hid (from x and the neighbors)
        self.spratio = spratio

        if self.spratio <= 0:
            if absin == 5:
                self.promptff = funcfenc(2 * n_in, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptfin = funcfenc(2 * n_h1, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptf = funcfenc(2 * n_h2, self.bottleout, nonl=None, bias=True, hid=self.hid)

            elif absin == 0:
                self.promptff = funcfenc(n_in, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptfin = funcfenc(n_h1, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptf = funcfenc(n_h2, self.bottleout, nonl=None, bias=True, hid=self.hid)

        else:
            if absin == 5:
                w_m1 = nn.Parameter(torch.empty(2 * n_h1, hid))
                nn.init.kaiming_uniform_(w_m1, a=math.sqrt(5))
                self.score1 = w_m1

                w_m2 = nn.Parameter(torch.empty(2 * n_h2, hid))
                nn.init.kaiming_uniform_(w_m2, a=math.sqrt(5))
                self.score2 = w_m2

                self.promptff = funcfenc(2 * n_in, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptfin = funcfenc(2 * n_h1, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptf = funcfenc(2 * n_h2, self.bottleout, nonl=None, bias=True, hid=self.hid)

            elif absin == 0:
                w_m1 = nn.Parameter(torch.empty(n_h1, hid))
                nn.init.kaiming_uniform_(w_m1, a=math.sqrt(5))
                self.score1 = w_m1

                w_m2 = nn.Parameter(torch.empty(n_h2, hid))
                nn.init.kaiming_uniform_(w_m2, a=math.sqrt(5))
                self.score2 = w_m2

                # print(self.score1.shape, self.score2.shape)

                self.promptff = funcfenc(n_in, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptfin = funcfenc(n_h1, self.bottleout, nonl=None, bias=True, hid=self.hid)
                self.promptf = funcfenc(n_h2, self.bottleout, nonl=None, bias=True, hid=self.hid)

        self.absin = absin

        self.promptg = funcg(n_h2, n_h1, nonl=None, bias=False)
        # todo by default use two mlp for extractions
        self.cls = nn.Linear(2 * self.hid, self.n_way, bias=True)

        self.ladj = ladj
        self.madj = madj

        if pemb is not None:
            self.pemb = pemb
        else:
            self.pemb = None

    def getdim(self):
        return self.outdim

    def forward(self, features, seq1, adj, vars, sparse=True, bias=True, train=True, unbias=False, ma_et1=1, ma_et2=1):

        if vars is None:
            vars = [j for i, j in self.named_parameters()]
        else:
            vars = [i for i in vars]
            # print(vars[0])
        # print([i.shape for i in vars])
        # print('vars', vars)
        # print('var len: ', len(vars), [i.shape for i in vars], seq1)
        # bias, weight, activation
        # print(vars[1].shape, seq1.shape)
        # print(vars[1].T.shape)

        if self.pemb is None:
            print('Costing: dgims is forwarding with spmm')
            ind = 0
            output = [features]

            for i in range(self.layer):
                if i == 0:
                    h1 = torch.matmul(output[-1], vars[ind + 1].T)
                else:
                    h1 = torch.matmul(output[-1], vars[ind + 1].T)
                if sparse:
                    h1 = torch.spmm(adj, torch.squeeze(h1, 0))
                else:
                    h1 = torch.bmm(adj, h1)

                if bias is True:
                    h1 = h1 + vars[ind]

                h1 = F.prelu(h1, vars[ind + 2])
                ind += 3
                output.append(h1)

            # mfl = mf[seq1]
            return None, 0
        else:
            if self.absin == 5:
                hl = self.pemb[-1][seq1]
                hinl = self.pemb[-2][seq1]
                f0l = self.pemb[-3][seq1]
                habsf0l = self.pemb[0][seq1]
                habsinl = self.pemb[1][seq1]
                habslastl = self.pemb[2][seq1]

                # todo use add #
                # augin = hinl + habsinl
                # auglast = hl + habslastl
                # augf0 = f0l + habsf0l

                # todo no aug
                #augin = hinl
                #auglast = hl
                #augf0 = f0l
                # print('cat')
                # todo use cat
                augf0 = torch.cat((f0l, habsf0l), -1)
                augin = torch.cat((hinl, habsinl), -1)
                auglast = torch.cat((hl, habslastl), -1)

                # print(vars[-15].shape, vars[-14].shape,vars[-13].shape, vars[-12].shape,vars[-11].shape,
                # vars[-10].shape, vars[-9].shape, vars[-8].shape, vars[-7].shape, vars[-6].shape, vars[-5].shape,
                # vars[-4].shape, vars[-3].shape)

                # mfl, mfinl, mfmidl, mfinmidl = aggrmwPure2l(auglast, augin,  self.ladj, self.madj,
                # vars[-11], vars[-10], vars[-9], vars[-8], vars[-7], vars[-6], vars[-5], vars[-4], vars[-3], train)
                if self.spratio <= 0:
                    mfl, mfinl, mf0l, mfmidl, mfinmidl, mf0midl = aggrmwPure3l(auglast, augin, augf0, self.ladj,
                                                                           self.madj,
                                                                           vars[-15], vars[-14], vars[-13],
                                                                           vars[-12],
                                                                           vars[-11], vars[-10], vars[-9], vars[-8],
                                                                           vars[-7], vars[-6], vars[-5], vars[-4],
                                                                           vars[-3], train)

                else:
                    mfl, mfinl, mf0l, mfmidl, mfinmidl, mf0midl = aggrmwPure3lSP(auglast, augin,  augf0, self.ladj, self.madj,
                                                                           vars[-15], vars[-14],vars[-13], vars[-12],
                                                                           vars[-11], vars[-10], vars[-9], vars[-8],
                                                                           vars[-7], vars[-6], vars[-5], vars[-4],
                                                                           vars[-3], train=train, sc1=vars[0],
                                                                        sc2=vars[1], ratio= self.spratio)

                # hdout, hin_out, hf0_out, hdmid, hin_mid, hf0_mid
                # concat = torch.cat((mfl, mfinl), dim=-1)  # out concat
                concat = torch.cat((mfmidl, mfinmidl), dim=-1)  # mid concat
                # concat = mfmidl

                # todo discarded lossm
                # lossm = torch.norm(mfmidl - mfinmidl, dim=-1)
                # distance = constr(mfmidl, mfinmidl)
                # lossm = torch.mean(lossm, dim=0)

                # todo current loss mi
                miloss1, ma_et1 = self.prompt_mi1(hinl, mfinmidl, ma_et1)
                miloss2, ma_et2  = self.prompt_mi2(hl, mfmidl, ma_et2)
                miloss = miloss1 + miloss2

                # miloss = max(0, miloss1) + max(0, miloss2)
                # miloss = max(miloss, -miloss)

            else:
                f0l = self.pemb[-3][seq1]
                hl = self.pemb[-1][seq1]
                hinl = self.pemb[-2][seq1]
                if self.spratio <= 0:

                    mfl, mfinl, mf0l, mfmidl, mfinmidl, mf0midl = aggrmwPure3l(hl, hinl, f0l,  self.ladj, self.madj,
                                                                           vars[-15], vars[-14], vars[-13], vars[-12],
                                                                           vars[-11], vars[-10], vars[-9], vars[-8],
                                                                           vars[-7], vars[-6], vars[-5], vars[-4],
                                                                           vars[-3], train)
                else:
                    mfl, mfinl, mf0l, mfmidl, mfinmidl, mf0midl = aggrmwPure3lSP(hl, hinl, f0l, self.ladj, self.madj,
                                                                               vars[-15], vars[-14], vars[-13],
                                                                               vars[-12],
                                                                               vars[-11], vars[-10], vars[-9], vars[-8],
                                                                               vars[-7], vars[-6], vars[-5], vars[-4],
                                                                               vars[-3], train=train, sc1=vars[0],
                                                                                 sc2=vars[1], ratio= self.spratio)

                concat = torch.cat((mfmidl, mfinmidl), dim=-1)
                # concat = mfmidl

                # todo discarded lossm
                # lossm = torch.norm(mfmidl - mfinmidl, dim=-1)
                # lossm = torch.mean(lossm, dim=0)

                # todo current loss mi
                # print(hl.shape, concat.shape)
                miloss1, ma_et1 = self.prompt_mi1(hinl, mfinmidl, ma_et1)
                miloss2, ma_et2 = self.prompt_mi2(hl, mfmidl, ma_et2)
                # miloss1 = self.prompt_mi1(hinl, mfinmidl)
                # miloss2 = self.prompt_mi2(hl, mfmidl)
                miloss = miloss1 + miloss2
                # miloss = max(0, miloss1) + max(0, miloss2)
                # miloss = max(miloss, -miloss)

            assert self.ladj is None
            logits = torch.matmul(concat, vars[-2].T) + vars[-1]
            # print(miloss)
            return logits, miloss

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_l1 = self.gcn1(seq, adj, sparse)
        h_l2 = self.gcn2(h_l1, adj, sparse)

        return seq.detach(), h_l1.detach(), h_l2.detach()  # c.detach()

    def get_mask(self, kth, layer_score):
        mone = torch.ones(layer_score.size())
        mzero = torch.zeros(layer_score.size())

        def percentile(t, q):
            k = 1 + round(.01 * float(q) * (t.numel() - 1))
            return t.view(-1).kthvalue(k).values.item()

        kth_value = torch.FloatTensor([percentile(layer_score, kth)])
        out = torch.where(layer_score < kth_value, mzero, mone)

        return out