import torch
from torch import nn


class EpsilonGreedy(nn.Module):
    def __init__(self, network, epsilon=0.05):
        super().__init__()
        self.network = network
        self.epsilon = epsilon

    def forward(self, x):
        out = self.network(x)
        prob = torch.rand(1)
        if prob > self.epsilon:
            action = torch.argmax(out)
        else:
            sample_idx = torch.randint(out.shape[0], size=(1,))
            action = out[sample_idx]
        return action.to(torch.int32)
