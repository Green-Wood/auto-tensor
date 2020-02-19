from typing import List

from tensor import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def step(self):
        """
        take a little step forward gradient
        :return: None
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, params: List[Tensor], lr: float):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for ts in self.params:
            ts.data -= self.lr * ts.grad.data
