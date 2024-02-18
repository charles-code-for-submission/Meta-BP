import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import optim
from meta_gnn.utils import f1
from colorama import Fore
from meta_gnn.mine.models.mine import get_est


class Meta(nn.Module):
    def __init__(self, args, encoder=None, config=None, ladj=None, madj=None, frozen=None, pemb=None, bottleout=None, absin=0, spratio=0):
        super(Meta, self).__init__()

        if args.b2 > 0:
            from dgi.models import DGI2ms2l_mi_lth_2b as DGIms2l
        else:

            from dgi.models import DGI2ms2l_mi_lth as DGIms2l

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        if frozen is None:
            print(Fore.RED, 'Error: frozen code unclear')
        self.frozencode = frozen
        self.pweight = args.pweight
        self.spratio = spratio

        if encoder is None:
            model = config['name']
            assert model == 'dgi'
            hin, hid_units = config['hid']
            nonl = config['nonlinear']
            hid = args.enchid
            if frozen == 1:
                self.net = DGIms2l(hin, hid_units[0], hid_units[1], self.n_way, nonl, ladj, madj, hid, pemb, bottleout,
                                   absin, spratio=self.spratio)
            else:
                print('Error no base model in meta')

        else:
            self.net = encoder

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def filter(self, params):

        if self.frozencode == 0:
            return [i for i in params]
        elif self.frozencode == 1:
            return [i for i in params][-15:]
        elif self.frozencode == 2:
            return [i for i in params][-2:]

    def nonfilter(self, params1, ori):
        ori1 = [i for i in ori]
        params11 = [i for i in params1]

        if self.frozencode == 0:
            return params11
        elif self.frozencode == 1:
            return ori1[:-15] + params11[-15:]
        elif self.frozencode == 2:
            return ori1[:-2] + params11[-2:]

    def forward(self, step, features,  x_spt, y_spt, x_qry, y_qry, adj, trainflag):
        task_num = self.task_num
        querysz = self.n_way * self.k_qry

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        losses_qm = [0 for _ in range(self.update_step + 1)]
        for i in range(task_num):

            logits, loss_m = self.net(features, x_spt[i], adj, vars=None, train=trainflag)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.filter(self.net.parameters()), allow_unused=True)

            updated = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad, self.filter(self.net.parameters()))))
            fast_weights = self.nonfilter(updated, self.net.parameters())
            with torch.no_grad():

                logits_q, loss_qm = self.net(features, x_qry[i], adj, self.net.parameters(), train=trainflag)

                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                losses_qm[0] += loss_qm
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            if self.update_step > 1:
                with torch.no_grad():
                    logits_q, loss_qm = self.net(features, x_qry[i], adj, fast_weights, train=trainflag)
                    loss_q = F.cross_entropy(logits_q, y_qry[i])
                    losses_q[1] += loss_q
                    losses_qm[1] += loss_qm

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct
            else:
                logits_q, loss_qm = self.net(features, x_qry[i], adj, fast_weights, train=trainflag)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                losses_qm[1] += loss_qm
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[ 1] = corrects[ 1] + correct

            for k in range(1, self.update_step):
                logits, loss_m = self.net(features, x_spt[i], adj, fast_weights, train=trainflag)
                loss = F.cross_entropy(logits, y_spt[i])

                grad = torch.autograd.grad(loss, self.filter(fast_weights), allow_unused=True)

                updated = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad, self.filter(fast_weights))))
                fast_weights = self.nonfilter(updated, fast_weights)
                logits_q, loss_qm = self.net(features, x_qry[i], adj, fast_weights, train=trainflag)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                losses_qm[k + 1] += loss_qm

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_qt = losses_q[-1] / task_num
        loss_qmt = losses_qm[-1] / task_num
        self.meta_optim.zero_grad()
        (loss_qt + self.pweight * loss_qmt).backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)

        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return_loss = True
        if return_loss is True:
            loss_qout = torch.stack(losses_q, 0).detach().cpu().numpy() / task_num
            return accs, loss_qout, loss_qmt
        else:
            return accs

    def forward_f1(self, x_spt, y_spt, x_qry, y_qry):
        task_num = self.task_num
        querysz = self.n_way * self.k_qry

        losses_q = [0 for _ in range(self.update_step + 1)]
        f1s = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                result = f1(logits_q, y_qry[i])
                f1s[0] = f1s[0] + result

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                result = f1(logits_q, y_qry[i])
                f1s[1] = f1s[1] + result

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    result = f1(logits_q, y_qry[i])
                    f1s[k + 1] = f1s[k + 1] + result

        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(f1s) / (querysz * task_num)
        return accs
