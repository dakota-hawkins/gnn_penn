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


class TwoLayerNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.parameter.Parameter(torch.rand(input_size, hidden_size))
        self.layer2 = nn.parameter.Parameter(torch.rand(hidden_size, output_size))

    @property
    def layer1(self) -> nn.parameter.Parameter:
        return self.layer1_

    @layer1.setter
    def layer1(self, value: nn.parameter.Parameter):
        if not isinstance(value, nn.parameter.Parameter):
            raise ValueError(f"Expected Parameter class, got: {type(value)}")
        self.layer1_ = value

    @property
    def layer2(self) -> nn.parameter.Parameter:
        return self.layer2_

    @layer2.setter
    def layer2(self, value: nn.parameter.Parameter):
        if not isinstance(value, nn.parameter.Parameter):
            raise ValueError(f"Expected Parameter class, got: {type(value)}")
        self.layer2_ = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = nn.ReLU()
        y_hat = torch.matmul(activation(torch.matmul(x, self.layer1)), self.layer2)
        return y_hat
