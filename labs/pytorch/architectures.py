import torch
import torch.nn as nn


class Parametrization(nn.Module):
    def __init__(self, n_rows: int, n_cols: int):
        super().__init__()
        self.H = nn.parameter.Parameter(torch.zeros(n_rows, n_cols))

    @property
    def H(self):
        return self.H_

    @H.setter
    def H(self, value: nn.Parameter):
        if not isinstance(value, nn.Parameter):
            raise ValueError(f"Expected torch parameters, got {type(value)}")
        self.H_ = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.H)  # expects x ~ (p x n), H ~ (n x m)
        activation = nn.ReLU()
        return activation(z)
