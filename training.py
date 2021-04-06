import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class Loss(nn.Module):
  def __init__(self):
    super(Loss, self).__init__()
    self.criterion = nn.NLLLoss()

  def forward(self, pred, tgt):
    return criterion(torch.log(pred), tgt)

"""# Optimizer"""

class Dynamic_LR_Scheduler():
    def __init__(self, d_model, beta1, beta2, epsilon, warmup_step, optimizer, factor = 1):
        super(Dynamic_LR_Scheduler, self).__init__()
        self.d_model= d_model
        self.beta1= beta1
        self.beta2= beta2
        self.epsilon = epsilon
        self.warmup_step = warmup_step
        self.optimizer = optimizer
        self.step_num = 0
        self.lr = 0
        self.factor = factor
        
    def step(self):
        self.step_num += 1
        lr = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.lr = lr
        return self.optimizer.step()
        
    def learning_rate(self):
        return self.factor * \
            (self.d_model ** (-0.5) *
            min(self.step_num ** (-0.5), self.step_num * self.warmup_step ** (-1.5)))

"""# Label Smoothing"""
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean") # reduction should be batch mean
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # print(true_dist)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
