import torch.nn as nn


class ShelterOutcomeModel(nn.Module):
    def __init__(self):
        super(ShelterOutcomeModel, self).__init__()
        self.lin1 = nn.Linear(5, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))

        return x