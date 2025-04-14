import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    layers: list = field(default_factory=lambda: [512,256,128,64])
    n_inputs: int = 6
    n_outputs: int = 1

class Regressor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.inputs = nn.Sequential(
            nn.Linear(config.n_inputs, config.layers[0]),
            nn.ReLU(),
        )

        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(config.layers[i], config.layers[i+1]),
                nn.ReLU(),
             )  for i in range(len(config.layers) - 1)]
        )

        self.outputs = nn.Sequential(
            nn.Linear(config.layers[-1], config.n_outputs)
        )

    def forward(self, x, targets=None):
        x = self.inputs(x)

        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.outputs(x)

        loss = None
        if targets is not None:
            loss = F.mse_loss(logits, targets)

        return logits, loss