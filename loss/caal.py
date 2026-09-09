"""
CAAL - Class-Adaptive Asymmetric Loss.

Adapted from:
    Luo, M., Min, W., Wang, Z., Song, J., & Jiang, S. (2023).
    "Ingredient Prediction via Context Learning Network With Class-Adaptive
    Asymmetric Loss." IEEE Transactions on Image Processing, 32, 5509-5523.
    doi:10.1109/TIP.2023.3318958

CAAL extends ASL (Ben-Baruch et al., 2020) with a per-class negative focus
parameter, adjusted by the accumulated positive-to-negative gradient ratio
using the gradient-guided reweighting of EQLv2 (Tan et al., CVPR 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class CAAL(nn.Module):
    def __init__(self,
                 num_classes, 
                 focal_gamma, 
                 scale_factor,
                 gamma,      
                 mu,       
                 alpha):
      
        super(CAAL, self).__init__()

        self.focal_gamma = focal_gamma
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.num_classes = num_classes

        self.scale_factor = scale_factor
        self.n_i = 0
        self.n_c = 0

        # used to store the +ve gradients for each class
        self._pos_grad = None

        # used to store the -ve gradients for each class
        self._neg_grad = None
        self.pos_neg = torch.tensor(1)

        # sigmoid function with gamma and nu
        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func,  gamma=self.gamma, mu=self.mu)

        self.clip = 0.05
        self.gamma_pos =  torch.tensor(2).cuda()
        self.dy_gamma =   torch.tensor(self.focal_gamma).cuda() 
       
        
    def reweight_functions(self, label):
        weight = self.rebalance_weight(label.float()) 
        weight_pos = weight * label
        weight_neg = weight * (1 - label)
        max_by_instance_pos, _ = torch.max(weight_pos, dim=-1, keepdim=True)
        weight_pos = weight_pos / max_by_instance_pos
        max_by_instance_neg, _ = torch.max(weight_neg, dim=-1, keepdim=True)
        weight_neg = weight_neg / max_by_instance_neg
        weight = weight_pos + weight_neg
        return weight

    def rebalance_weight(self, gt_labels): 
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True) 
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate 
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha 
        return weight

    def forward(self, input, target):
        
        # batch size and num classes
        self.n_i, self.n_c = input.size()
        targets = target
        pos_w, neg_w = self.get_weight(input)

        # calculation of gradients using weights
        weight = pos_w * target + neg_w * (1 - target)

        # getting gradient ratio
        self.collect_grad(input.detach(), target.detach(), weight.detach())
        map_val = 1 - self.pos_neg.detach() 

        # altering neg focusing parameter based on ratio
        self.dy_gamma = self.focal_gamma + self.scale_factor * map_val
        pred = torch.sigmoid(input)
        xs_pos = pred
        xs_neg = 1 - pred
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8)) 
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))  
        loss = -(los_pos + los_neg)
    
        torch._C._set_grad_enabled(False)
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)  
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * targets + self.dy_gamma.cuda().detach() * (1 - targets)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        torch._C._set_grad_enabled(True)
        cls_loss = loss * one_sided_w  

        return cls_loss.sum() 

    def collect_grad(self, cls_score, target, weight):
        # yielding sigmoid of output
        prob = torch.sigmoid(cls_score)

        # positive and negative values
        xs_pos = prob
        xs_neg = 1 - prob

        # probability shifting like in ASL
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # constant pos gamma and dynamic neg gamma focus params
        ga_pos = self.gamma_pos.cuda()
        ga_neg = self.dy_gamma.cuda()

        # positive and negative equations
        grad_p = target * ga_pos * xs_pos * torch.log(xs_pos.clamp(min=1e-10)) * torch.pow(xs_neg, ga_pos) - target * torch.pow(xs_neg, ga_pos) * (xs_neg)
        grad_n = (target - 1) * ga_neg * torch.pow(xs_pos, ga_neg) * (xs_neg) * torch.log((xs_neg).clamp(min=1e-10)) + (1 - target) * torch.pow(xs_pos, ga_neg + 1)
        grad = grad_p + grad_n  
        grad = torch.abs(grad)

        # gradient eqn of pos and neg + accumuation across iterations
        pos_grad = torch.sum(grad * target * weight, dim=0) 
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)
        self._pos_grad += pos_grad
        self._neg_grad += neg_grad

        # calculation of pos neg grad ratio
        self.pos_neg = torch.clamp(self._pos_grad / (self._neg_grad + 1e-10), min=0, max=1)  

    def get_weight(self, cls_score):
        if self._pos_grad is None:

            # initialises pos and neg grad to store zeros for grad values
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)

            # stores ones for pos and neg weights for each class
            neg_w = cls_score.new_ones((self.n_i, self.n_c))
            pos_w = cls_score.new_ones((self.n_i, self.n_c))
        else:
            # applied sigmoid on pos_neg ratio
            neg_w = self.map_func(self.pos_neg)
            alpha = self.alpha

            # calculates pos weight using neg_w and alpha
            pos_w = 1 + alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


