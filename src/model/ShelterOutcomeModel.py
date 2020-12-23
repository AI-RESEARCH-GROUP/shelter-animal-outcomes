import torch
import torch.nn as nn
import torch.nn.functional as F


class ShelterOutcomeModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear()
        self.lin3 = nn.Linear()
        self.bn1 = nn.BatchNorm1d()
        self.bn2 = nn.BatchNorm1d()
        self.bn3 = nn.BatchNorm1d()
        self.drop = nn.Dropout()

    def forward():

        return