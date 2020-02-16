from typing import List, Set
import numpy as np


class Tensor:
    # We can view a Tensor as a Binary Tree
    def __init__(self,
                 data: np.ndarray,
                 name: str,
                 requires_grad: bool = False,
                 lhs=None,
                 rhs=None,
                 operation=None):
        self.data = data
        self.name = name
        self.requires_grad = requires_grad
        self.lhs: Tensor = lhs
        self.rhs: Tensor = rhs
        self.operation: Operation = operation
        self.grad: Tensor = None

    def backward(self):
        """start backpropagation from current tensor, accumulate to each tensor's gradient"""
        from collections import deque

        self.grad = ones_like(self, None)
        queue = deque([self, ])
        # using hierarchy traversal
        while queue:
            ts = queue.popleft()
            ts.operation.backward(ts.lhs, ts.rhs, ts.grad)
            if ts.lhs and ts.lhs.operation:
                queue.append(ts.lhs)
            if ts.rhs and ts.rhs.operation:
                queue.append(ts.rhs)

    def __add__(self, other):

        if not isinstance(other, Tensor):
            # check whether it is a scala, List or Tensor
            np_data = np.array(other)
            other = Tensor(np_data, str(other))
        return add(self, other)

    __radd__ = __add__


class Operation:

    def __call__(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return self.forward(lhs, rhs)

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """
        calculate forward new Tensor
        :param lhs: left hand operator, used when there is only one operator
        :param rhs: right hand operator, None when there is only one operator
        :return:
        """
        raise NotImplementedError

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        """
        calculate backward new Tensor gradient
        :param rhs: left hand operator, used when there is only one operator
        :param lhs: right hand operator, None when there is only one operator
        :param acc_grad: accumulated gradient until now
        :return: None
        """
        raise NotImplementedError


class AddOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data + rhs.data
        new_name = '({}+{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        lhs.grad = lhs.grad + acc_grad if lhs.grad else acc_grad
        rhs.grad = rhs.grad + acc_grad if rhs.grad else acc_grad


class ZerosLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.zeros_like(lhs.data)
        new_name = 'ZerosLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        lhs.grad = zeros_like(lhs, None)


class OnesLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.ones_like(lhs.data)
        new_name = 'OnesLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        lhs.grad = zeros_like(lhs, None)


def tensor(data, name: str, requires_grad: bool = False) -> Tensor:
    """Create Tensor user friendly"""
    return Tensor(np.array(data), name, requires_grad)


# singleton factory
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
add = AddOp()
