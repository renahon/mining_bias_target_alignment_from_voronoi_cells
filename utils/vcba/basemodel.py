# File:                     basemodel.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# BaseModel class used for Bmnist for the biased and the debiased models
# ================================= IMPORTS ==================================
from utils.tools import *
from utils.configs import *
from utils.biased_mnist import SimpleConvNet
from utils.vcba.information_removal import MI, PrivacyHead
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from utils.vcba.reweighting import VCBAReweighter

# =================================== CODE ====================================


class VCBABaseModel:
    def __init__(
        self, device: torch.device, learning_params: dict, dataset_params: dict
    ):
        self.dataset = dataset_params["dataset"]
        self.epochs = learning_params["epochs"]
        self.device = device
        self._init_model(learning_params)
        self._init_optimizer()
        self._init_scheduler()
        self.criterion = CrossEntropyLoss(
            reduction=learning_params["criterion_reduction"]
        ).to(self.device)
        self.bottleneck = Hook(self.model.avgpool, backward=False)
        self.reweighter = VCBAReweighter(
            nb_features=self.nb_features,
            target_classes=self.target_classes,
            device=self.device,
            epochs=self.epochs,
            dataset_size=self.dataset_size,
            batch_size=learning_params["batch_size"],
            bias_inference_metric=learning_params["bias_inference_metric"],
        )

    def _init_model(self, learning_params: dict):
        self.weight_decay = learning_params["weight_decay"]
        self.lr = learning_params["lr"]
        self.momentum_sgd = learning_params["momentum_sgd"]
        self.nb_features = 128
        self.nb_classes = 10
        self.dataset_size = 60000
        self.target_classes = torch.arange(0, 10, dtype=int)
        self.model = SimpleConvNet(num_classes=10).to(self.device)
        self.model.avgpool = nn.Sequential(
            self.model.avgpool, torch.nn.Identity().to(self.device)
        )

    def _init_optimizer(self):
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum_sgd,
            weight_decay=self.weight_decay,
        )

    def _init_scheduler(self):
        self.sched = MultiStepLR(
            self.optimizer, milestones=[40, 60], gamma=0.1, verbose=False
        )

    def set_data_loaders(self, dataloaders: dict):
        self.dataloaders = dataloaders

    def set_information_removal(self, gamma: int, ph_learning_params: dict):
        self.MI = MI(device=self.device, privates=self.nb_classes)
        self.PH = PrivacyHead(
            self.model.avgpool,
            nn.Sequential(torch.nn.Linear(self.nb_features, self.nb_classes)),
        ).to(self.device)
        self.PH_optimizer = SGD(
            self.PH.parameters(),
            lr=ph_learning_params["lr"],
            momentum=ph_learning_params["momentum_sgd"],
            weight_decay=ph_learning_params["weight_decay"],
        )
        self.gamma = gamma
