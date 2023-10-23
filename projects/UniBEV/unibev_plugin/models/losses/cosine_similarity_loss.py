import torch
from torch import nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import LOSSES


@LOSSES.register_module()
class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8, loss_weight=1.0):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, feat1, feat2):
        feat2 = feat2.flatten(2).permute(0, 2, 1)
        assert feat1.size() == feat2.size()

        loss_cosine_similarity = (1 - F.cosine_similarity(feat1, feat2, dim=-1, eps=self.eps)) * self.loss_weight

        loss_cosine_similarity = torch.mean(loss_cosine_similarity, dim=-1)
        return loss_cosine_similarity