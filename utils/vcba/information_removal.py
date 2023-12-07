# File:                     information_removal.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# Implementation of the Information Removal Module
# inspired by https://github.com/enzotarta/irene/blob/main/irene/core.py
#
# ================================= IMPORTS =================================
import torch
from torch.nn.functional import one_hot
import numpy as np
from utils.tools import Hook

# ================================== CODE ===================================


class PrivacyHead(torch.nn.Module):
    def __init__(self, bottleneck_layer, head_structure):
        super(PrivacyHead, self).__init__()
        self.bottleneck = Hook(bottleneck_layer, backward=False)
        self.classifier = head_structure

    def forward(self):
        x = self.bottleneck.output.clone().detach()
        if len(x.size()) > 2:
            x = x.view(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x

    def forward_attached(self):
        x = self.bottleneck.output
        if len(x.size()) > 2:
            x = x.view(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x


class MI(torch.nn.Module):
    def __init__(self, privates: int = 10, device: str = "cpu"):
        super(MI, self).__init__()
        self.device = device
        self.privates = privates
        self.scaling = 1 / np.log(privates)

    def _get_joint_and_marginals(self, GT_private_onehot, prob_private, nb_samples):
        joint = (
            torch.clamp(
                torch.mm(torch.transpose(GT_private_onehot, 0, 1), prob_private),
                min=1e-15,
            )
            / nb_samples
        )
        marginals_out_private = torch.sum(joint, dim=0, keepdim=True)
        marginals_GT_private = torch.sum(joint, dim=1, keepdim=True)
        marginals = torch.clamp(
            torch.mm(marginals_GT_private, marginals_out_private), min=1e-15
        )
        return joint, marginals

    def forward(self, private_head, private_label, mask):
        out_private = private_head.forward_attached()
        GT_private_onehot = 1.0 * one_hot(private_label, num_classes=self.privates)
        prob_private = torch.nn.functional.softmax(out_private, dim=1)
        GT_private_onehot_masked = GT_private_onehot[mask, :]
        prob_private_masked = prob_private[mask, :]
        nb_samples_masked = torch.sum(GT_private_onehot_masked)
        (joint_masked, marginals_masked) = self._get_joint_and_marginals(
            GT_private_onehot_masked, prob_private_masked, nb_samples_masked
        )
        return torch.sum(
            joint_masked * torch.log(joint_masked / marginals_masked) * self.scaling
        )
