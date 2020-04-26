import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNConv1D(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()

        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=shape[0], out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
            nn.ReLU(),
        )

        # Get the output size of the convolutional layers
        # to give to the FC layers below
        out_size = self._get_out_size(shape)

        # The fully connected value network
        self.fc_val = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )
        # The fully connected advantage network
        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def _get_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_output = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_output)
        adv = self.fc_adv(conv_output)
        # Combine the values of the value and advantage
        # networks to get final estimate per action
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1DLarge1(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()
        
        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=shape[0], out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Get the output size of the convolutional layers
        # to give to the FC layers below
        out_size = self._get_conv_out(shape)

        # The fully connected value network
        self.fc_val = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(in_features=512, out_features=1)
        )

        # The fully connected advantage network
        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        # Combine the values of the value and advantage
        # networks to get final estimate per action
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1DLarge2(nn.Module):
    def __init__(self, shape, n_actions):
        super().__init__()

        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=shape[0], out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Get the output size of the convolutional layers
        # to give to the FC layers below
        out_size = self._get_conv_out(shape)

        # The fully connected value network
        self.fc_val = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=1)
        )

        # The fully connected advantage network
        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=out_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        # Combine the values of the value and advantage
        # networks to get final estimate per action
        return val + adv - adv.mean(dim=1, keepdim=True)
