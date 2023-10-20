import typing
import numpy as np
from numpy.typing import NDArray
import torch
import torch.optim as optim

from architectures import Parametrization, TwoLayerNN


class LinearFunction:
    def __init__(self, n_rows: int, n_cols: int, A: NDArray):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.A = A

    def __repr__(self, name: str = "LinearFunction"):
        return f"{name} with {self.n_rows}x{self.n_cols} operating matrix."

    @property
    def n_rows(self):
        """Number of rows in weight matrix"""
        return self.n_rows_

    @n_rows.setter
    def n_rows(self, value: int):
        if int(value) != float(value):
            raise ValueError(f"Expected integer for `n_rows`, got {type(value)}")
        self.n_rows_ = value

    @property
    def n_cols(self):
        """Number of columns in weight matrix"""
        return self.n_cols_

    @n_cols.setter
    def n_cols(self, value: int):
        if int(value) != float(value):
            raise ValueError(f"Expected integer for `n_cols`, got {type(value)}")
        self.n_cols_ = value

    @property
    def A(self):
        """Linear translation matrix."""
        return self.A_

    @A.setter
    def A(self, value: NDArray):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Expected array for `A`, got {type(value)}")
        if value.shape != (self.n_rows, self.n_cols):
            raise ValueError(
                f"Provided matrix does not matched expected shape: ({self.n_rows}, {self.n_cols}). "
                f"Received {value.shape}"
            )
        self.A_ = value

    def evaluate(self, x: NDArray) -> NDArray:
        """Evaluate linear map

        Args:
            x (NDArray): matrix to apply linear map to

        Returns:
            NDArray: predicted output
        """
        assert x.shape[0] == self.n_cols
        return self.A @ x


class LinearBernoulli(LinearFunction):
    def __init__(self, n_rows: int, n_cols: int):
        A = np.random.binomial(n=1, p=0.5, size=(m, n))
        super().__init__(n_rows, n_cols, A)

    def __repr__(self):
        return super().__repr__("LinearBernoulli")


def get_batch(batch_size, X, y) -> (torch.Tensor, torch.Tensor):
    pass


def train(
    X: NDArray,
    y: NDArray,
    estimator: torch.nn.Parameter,
    batch_size,
    n_iters: int = 100,
    eps: float = 10e-4,
):
    iter = 0
    optimizer = optim.SGD(estimator.parameters(), lr=eps, momentum=0)

    while iter < n_iters:
        X_, y_ = get_batch(batch_size, X, y)
        estimator.zero_grad()
        y_hat = estimator.forward(X_)
        loss = torch.mean((y_hat - y) ** 2)
        loss.backward()
        optimizer.step()
        iter += 1


if __name__ == "__main__":
    m = 42
    n = 71
    A = np.random.binomial(n=1, p=0.5, size=(m, n))
    bernoulli_map = LinearFunction(m, n, A)
    print(bernoulli_map)
    x = np.random.binomial(n=1, p=0.5, size=(n, 1))
    y = bernoulli_map.evaluate(x)
    print(f"x: {x[:5].T}, y: {y[:5].T}")
    bernoulli_inhr = LinearBernoulli(m, n)
    print(bernoulli_inhr)
    print(f"x: {x[:5].T}, y: {bernoulli_inhr.evaluate(x)[:5].T}")

    h = 35
    two_layer_nn = TwoLayerNN(input_size=n, output_size=m, hidden_size=h)


# (m x n) x (n, 1)
