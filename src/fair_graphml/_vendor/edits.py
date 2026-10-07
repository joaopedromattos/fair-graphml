"""EDITS optimization core copied from pre_processing_methods/EDITS/EDITS.py.
CPU/dtype handling fixed; experiment loops and evaluation imports removed.
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.optimizer import required
import numpy as np

class PGD(Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable
        iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)

    """

    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        super(PGD, self).__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', [])
            group.setdefault('alphas', [])

    def step(self, delta=0, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']
            for param in group['params']:
                for (prox_operator, alpha) in zip(proxs, alphas):
                    param.data = prox_operator(param.data, alpha=alpha * lr)

class ProxOperators:

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm."""
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data) - alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm)."""
        device = data.device
        (U, S, V) = np.linalg.svd(data.cpu())
        (U, S, V) = (torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device))
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S - alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

prox_operators = ProxOperators()

class EDITS(nn.Module):

    def __init__(self, feat, lr=0.003, weight_decay=1e-07, nclass=2, adj_lambda=0.1, layer_threshold=2, dropout=0.1):
        super(EDITS, self).__init__()
        node_num = feat.shape[0]
        nfeat = feat.shape[-1]
        nfeat_out = nfeat // 10
        self.x_debaising = X_debaising(nfeat)
        self.layer_threshold = layer_threshold
        self.adj_renew = Adj_renew(node_num, nfeat, nfeat_out, adj_lambda)
        self.fc = nn.Linear(nfeat * (layer_threshold + 1), 1)
        self.lr = lr
        self.optimizer_feature_l1 = PGD(self.x_debaising.parameters(), proxs=[prox_operators.prox_l1], lr=self.lr, alphas=[5e-06])
        self.dropout = nn.Dropout(dropout)
        G_params = list(self.x_debaising.parameters())
        self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.lr, eps=0.0001, weight_decay=weight_decay)
        self.optimizer_A = torch.optim.RMSprop(self.fc.parameters(), lr=self.lr, eps=0.0001, weight_decay=weight_decay)

    def propagation_cat_new_filter(self, X_de, A_norm, layer_threshold):
        A_norm = A_norm
        X_agg = X_de
        for i in range(layer_threshold):
            X_de = A_norm.mm(X_de)
            X_agg = torch.cat((X_agg, X_de), dim=1)
        return X_agg

    def forward(self, A, X):
        X_de = self.x_debaising(X)
        adj_new = self.adj_renew()
        agg_con = self.propagation_cat_new_filter(X_de, adj_new, layer_threshold=self.layer_threshold)
        D_pre = self.fc(agg_con)
        D_pre = self.dropout(D_pre)
        return (adj_new, X_de, D_pre, D_pre, agg_con)

    def optimize(self, adj, features, idx_train, sens, epoch, lr, alpha=0.05, beta=0.1):
        """
        credit: alpha = 5e-2; beta = 0.1;
        german: alpha = 30e-2; beta = 0.8;
        bail: alpha = 3e-2; beta = 1;
        """



        self.lr = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_A.param_groups:
            param_group['lr'] = lr
        self.train()
        self.optimizer_G.zero_grad()
        self.optimizer_feature_l1.zero_grad()
        self.fc.requires_grad_(False)
        if epoch == 0:
            self.adj_renew.fit(adj, self.lr)
        (_, X_debiased, predictor_sens, show, agg_con) = self.forward(adj, features)
        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = -(torch.mean(positive_eles) - torch.mean(negative_eles))
        loss_train = alpha * (X_debiased - features).norm(2) + beta * adv_loss
        loss_train.backward()
        self.optimizer_G.step()
        self.optimizer_feature_l1.step()
        (_, X_debiased, predictor_sens, show, agg_con) = self.forward(adj, features)
        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = -(torch.mean(positive_eles) - torch.mean(negative_eles))
        self.adj_renew.train_adj(X_debiased, adj, adv_loss, epoch, lr)
        param = self.state_dict()
        zero = torch.zeros_like(param['x_debaising.s'])
        one = torch.ones_like(param['x_debaising.s'])
        param['x_debaising.s'] = torch.where(param['x_debaising.s'] > 1, one, param['x_debaising.s'])
        param['x_debaising.s'] = torch.where(param['x_debaising.s'] < 0, zero, param['x_debaising.s'])
        self.load_state_dict(param)
        for i in range(8):
            self.fc.requires_grad_(True)
            self.optimizer_A.zero_grad()
            (_, X_debiased, predictor_sens, show, agg_con) = self.forward(adj, features)
            positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
            negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
            loss_train = torch.mean(positive_eles) - torch.mean(negative_eles)
            loss_train.backward()
            self.optimizer_A.step()
            for p in self.fc.parameters():
                p.data.clamp_(-0.02, 0.02)

class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n), requires_grad=True)
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            adj = adj.to_dense()
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

class Adj_renew(nn.Module):

    def __init__(self, node_num, nfeat, nfeat_out, adj_lambda):
        super(Adj_renew, self).__init__()
        self.node_num = node_num
        self.nfeat = nfeat
        self.nfeat_out = nfeat_out
        self.adj_lambda = adj_lambda
        self.reset_parameters()

    def fit(self, adj, lr):
        estimator = EstimateAdj(adj, symmetric=False, device=adj.device).to(adj.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=lr)
        self.optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1], lr=lr, alphas=[0.0005])
        self.optimizer_nuclear = PGD(estimator.parameters(), proxs=[prox_operators.prox_nuclear], lr=lr, alphas=[1.5])

    def reset_parameters(self):
        pass

    def the_norm(self):
        return self.estimator._normalize(self.estimator.estimated_adj)

    def forward(self):
        return self.estimator.estimated_adj

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj
        r_inv = r_inv + 0.001
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

    def train_adj(self, features, adj, adv_loss, epoch, lr):
        for param_group in self.optimizer_adj.param_groups:
            param_group['lr'] = lr
        estimator = self.estimator
        estimator.train()
        self.optimizer_adj.zero_grad()
        delta = estimator.estimated_adj - adj
        loss_fro = torch.sum(delta.mul(delta))
        loss_diffiential = 1 * loss_fro + 15 * adv_loss
        loss_diffiential.backward()
        self.optimizer_adj.step()
        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()
        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))
        estimator.estimated_adj.data.copy_((estimator.estimated_adj.data + estimator.estimated_adj.data.transpose(0, 1)) / 2)
        return estimator.estimated_adj

class X_debaising(nn.Module):

    def __init__(self, in_features):
        super(X_debaising, self).__init__()
        self.in_features = in_features
        self.s = Parameter(torch.FloatTensor(in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.s.data.uniform_(1, 1)

    def forward(self, feature):
        return torch.mm(feature, torch.diag(self.s))
