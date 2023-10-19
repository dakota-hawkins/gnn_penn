import typing
import numpy as np
from numpy.typing import NDArray


class LinearFunction:
    def __init__(self, n_rows: int, n_cols: int, A: NDArray):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.A = self.A

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
    def n_rows(self, value: int):
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

    def evaluate(self, x: NDArray):
        assert x.shape[1] == self.n_rows
        return x @ self.A
