from typing import List

from tensor import Tensor
import numpy as np


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

    def __init__(self, params: List[Tensor], lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for ts in self.params:
            ts.data -= self.lr * ts.grad


class Adam(Optimizer):

    def __init__(self, params: List[Tensor], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.states = [(np.zeros_like(param.data), np.zeros_like(param.data)) for param in params]
        self.beta1_t = 1
        self.beta2_t = 1

    def step(self):

        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

        for ts, (m, v) in zip(self.params, self.states):
            m[:] = self.beta1 * m + (1 - self.beta1) * ts.grad
            v[:] = self.beta2 * v + (1 - self.beta2) * ts.grad ** 2
            m_hat = m / (1 - self.beta1_t)
            v_hat = v / (1 - self.beta2_t)
            ts.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


