# File:                     biased_mnist.py
# Created by:               ReBias - NAVER Corp
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
# Based on : ReBias
#            Copyright (c) 2020-present NAVER Corp.
#            MIT license
#            Python implementation of Biased-MNIST.
# => Modification : BiasedMnist.__getitem__()
# returns img, target, int(self.biased_targets[index]), index
#
# ================================= IMPORTS ==================================
import torch
from torch import nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils import data

### MOdif test
# =================================== CODE ===================================


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=None, kernel_size=7, feature_pos="post"):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ["pre", "post", "logits"]:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=True):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits

        elif self.feature_pos == "pre":
            feats = pre_gap_feats
        elif self.feature_pos == "post":
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats


class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.
    You have two paramters to control the level of bias.
    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.
        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).
        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.
    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [225, 225, 0],
        [225, 0, 225],
        [0, 255, 255],
        [255, 128, 0],
        [255, 0, 128],
        [128, 0, 255],
        [128, 128, 128],
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(
            np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels))
        )

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(
            indices[n_correlated_samples:], n_decorrelated_per_class
        )

        other_labels = [
            _label % 10
            for _label in range(label + 1, label + 1 + self.n_confusing_labels)
        ]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST."""
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target, int(self.biased_targets[index]), index)


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation=1.0,
        n_confusing_labels=9,
    ):
        super(ColourBiasedMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            data_label_correlation=data_label_correlation,
            n_confusing_labels=n_confusing_labels,
        )

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return (
            self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]),
            self.targets[indices],
        )


def get_biased_mnist_dataloader(
    root,
    batch_size,
    data_label_correlation,
    n_confusing_labels=9,
    train=True,
    num_workers=8,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = ColourBiasedMNIST(
        root,
        train=train,
        transform=transform,
        download=True,
        data_label_correlation=data_label_correlation,
        n_confusing_labels=n_confusing_labels,
    )
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
