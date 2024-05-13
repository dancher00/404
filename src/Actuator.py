dActuator.py

import torch
from torch import nn

class Actuator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.act1 = nn.Softsign()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.Softsign()
        self.fc3 = nn.Linear(32, 32)
        self.act3 = nn.Softsign()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.output(x)
        return x
