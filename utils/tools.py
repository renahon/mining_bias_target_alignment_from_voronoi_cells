# File:                     tools.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# ================================= IMPORTS =================================
from utils.biased_mnist import ColourBiasedMNIST
import torch.backends.cudnn as cudnn
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import os
import wandb

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ================================== CODE ===================================
__all__ = [
    "AverageMeter",
    "Hook",
    "Bottleneck",
    "set_seeds",
    "set_device",
    "build_dataloaders",
    "accuracy",
    "log_action",
    "add_dims_index"
]

# --------------------------------- Classes ---------------------------------


def print_remi(str_name, str):
    print(f"{str_name}={str}")


def add_dims_index(tensor: torch.Tensor, nb_dims: int, index: int):
    """
    """
    for i in range(nb_dims):
        tensor = tensor.unsqueeze(index)
    return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sq_sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sq_sum += n * val ** 2
        self.count += n
        #print(f"{self.name}: val = {self.val} sum = {self.sum} count= {self.count}")
        if self.count > 0:
            self.avg = self.sum / self.count
            self.std = (self.sq_sum/self.count - self.avg**2)**0.5

    def __str__(self):
        fmtstr = f"{self.name} {self.count} {self.avg} +/- {self.std}"
        return fmtstr


class Hook:
    """Registers a hook at a specific layer of a network"""

    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module[1].register_forward_hook(self.hook_fn)
            self.name = module[0]
        else:
            self.hook = module[1].register_backward_hook(self.hook_fn)
            self.name = module[0]

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Bottleneck(torch.nn.Module):
    def __init__(self, bottleneck_layer, device):
        super(Bottleneck, self).__init__()
        self.bottleneck = Hook(bottleneck_layer, backward=False)
        self.classifier = torch.nn.Identity(device=device)

    def forward(self):
        x = self.bottleneck.output.clone().detach()
        if len(x.size()) > 2:
            x = x.view(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x

    def forward_attached(self):
        # for the MI, not detached because
        # we want the mutual information to propagate back through the model
        x = self.bottleneck.output
        if len(x.size()) > 2:
            x = x.view(-1, np.prod((x.size())[1:]))
        x = self.classifier(x)
        return x


# -------------------------------- Functions --------------------------------
# Model Initialization


def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def set_device(args):
    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)


def build_dataloaders(dataset_params: dict):
    """
    Builds a dictionary containing the different dataloaders used
    """
    dataloaders = {}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = ColourBiasedMNIST(
        dataset_params["datapath"] + "MNIST/",
        train=True,
        download=True,
        data_label_correlation=dataset_params["rho"],
        n_confusing_labels=9,
        transform=transform,
    )
    test_dataset = ColourBiasedMNIST(
        dataset_params["datapath"] + "MNIST/",
        train=False,
        data_label_correlation=0.1,
        n_confusing_labels=9,
        transform=transform,
    )
    dataloaders["train"] = DataLoader(
        dataset=train_dataset,
        batch_size=dataset_params["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dataloaders["test"] = DataLoader(
        dataset=test_dataset,
        batch_size=dataset_params["test_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloaders


# General use


def accuracy(output, target, topk=(1,), num_classes=10):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if batch_size > 0:
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                if num_classes > 0:
                    res.append(1.0 / num_classes)
                else:
                    raise Exception("num_classes should be strictly positive")
        return res


def log_action(str_to_print: str, importance: int = 2):
    """
    Prints any kind of string with a specific frame regarding its importance
    """
    if importance == 1:
        print(f"{str_to_print}")
    elif importance == 2:
        print(f"===> {str_to_print}")
    elif importance == 3:
        print(f"====={str_to_print}=====")
