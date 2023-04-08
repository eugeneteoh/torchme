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
        # Only works when batch_size=1 now
        if prob > self.epsilon:
            return torch.argmax(out)
        sample_idx = torch.randint(out.shape[1], size=())
        return out[0, sample_idx]
