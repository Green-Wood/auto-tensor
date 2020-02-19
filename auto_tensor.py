from typing import Tuple

import numpy as np
from tensor import Tensor, tensor, ones, zeros
import nn
import optim


def accumulate_grad(target: Tensor, grad: Tensor):
    """
    Accumulate gradient to target Tensor
    :param target:
    :param grad:
    :return: None
    """
    # if this is a const, just return
    if target.is_const:
        return

    assert target.shape == grad.shape

    if target.grad:
        target.grad += grad
    else:
        target.grad = grad


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
        accumulate_grad(lhs, acc_grad)
        accumulate_grad(rhs, acc_grad)


class OnesLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.ones_like(lhs.data)
        new_name = 'OnesLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        pass


class ZerosLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.zeros_like(lhs.data)
        new_name = 'ZerosLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        pass


class MulOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data * rhs.data
        new_name = '({}*{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        accumulate_grad(lhs, rhs * acc_grad)
        accumulate_grad(rhs, lhs * acc_grad)


class DivOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data / rhs.data
        new_name = '({}/{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        numerator_grad = ones_like(lhs, None) / rhs
        denominator_grad = (-lhs) / (rhs * rhs)
        accumulate_grad(lhs, numerator_grad)
        accumulate_grad(rhs, denominator_grad)


class ExpOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.exp(lhs.data)
        new_name = 'exp({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        accumulate_grad(lhs, exp_op(lhs, None) * acc_grad)


class ViewOp(Operation):
    """Not a singleton, sacrifice performance to maintain interface consistency"""

    def __init__(self, new_shape: Tuple):
        self.new_shape = new_shape

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = lhs.data.reshape(self.new_shape)
        new_name = '(view({},{}))'.format(lhs.name, self.new_shape)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        acc_grad_reshape = view(acc_grad, lhs.shape)
        accumulate_grad(lhs, acc_grad_reshape)


class PermuteOp(Operation):

    def __init__(self, axes: Tuple):
        self.axes = axes

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.transpose(lhs.data, self.axes)
        new_name = '(permute({},{}))'.format(lhs.name, self.axes)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        new_axes = [0] * len(self.axes)
        # permute back to original space
        for k, v in enumerate(self.axes):
            new_axes[v] = k
        acc_grad_permute = permute(acc_grad, tuple(new_axes))
        accumulate_grad(lhs, acc_grad_permute)


class MatrixMulOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = np.matmul(lhs.data, rhs.data)
        new_name = 'matmul({},{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        lhs_trans = transpose(lhs)
        rhs_trans = transpose(rhs)
        accumulate_grad(lhs, matmul(acc_grad, rhs_trans))
        accumulate_grad(rhs, matmul(lhs_trans, acc_grad))


class CatOp(Operation):

    def __init__(self, axes: int):
        """
        cat two tensor with the same dim within given axes
        :param axes:
        """
        self.axes = axes

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert len(lhs.shape) == len(rhs.shape), 'Two tensor should have same dimension'

        new_data = np.concatenate((lhs.data, rhs.data), axis=self.axes)
        new_name = 'cat({},{},axes={})'.format(lhs.name, rhs.name, self.axes)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        lhs_len = lhs.shape[self.axes]
        rhs_len = rhs.shape[self.axes]
        lhs_grad, rhs_grad, _ = np.split(acc_grad.data, [lhs_len, lhs_len + rhs_len], self.axes)
        lhs_grad = Tensor(lhs_grad, '{}_grad'.format(lhs.name))
        rhs_grad = Tensor(rhs_grad, '{}_grad'.format(rhs.name))
        accumulate_grad(lhs, lhs_grad)
        accumulate_grad(rhs, rhs_grad)


class SumOp(Operation):

    def __init__(self, axes):
        self.axes = axes

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.sum(lhs.data, self.axes, keepdims=True)
        new_name = 'sum({},axis={})'.format(lhs.name, self.axes)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        repeat_len = lhs.shape[self.axes]
        acc_grad = np.repeat(acc_grad.data, repeat_len, axis=self.axes)
        acc_grad = Tensor(acc_grad, '{}_grad'.format(lhs.name))
        accumulate_grad(lhs, acc_grad)

class LogOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.log(lhs.data)
        new_name = 'log({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        accumulate_grad(lhs, 1 / lhs * acc_grad)


# singleton factory
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
add_op = AddOp()
mul_op = MulOp()
div_op = DivOp()
exp_op = ExpOp()
matmul = MatrixMulOp()
log_op = LogOp()


def exp(ts: Tensor) -> Tensor:
    """exp operation wrapper"""
    return exp_op(ts, None)

def log(ts: Tensor) -> Tensor:
    """log operation wrapper"""
    return log_op(ts, None)

def view(ts: Tensor, new_shape: Tuple) -> Tensor:
    view_op = ViewOp(new_shape)
    return view_op(ts, None)


def transpose(ts: Tensor) -> Tensor:
    """Quick func for 2 dim matrix"""
    assert len(ts.shape) == 2
    return permute(ts, (1, 0))


def permute(ts: Tensor, axes) -> Tensor:
    """
    Same as np.reshape, shape must be specified
    :param ts: tensor to permute
    :param axes: if None equals to transpose permute(1, 0)
    :return:
    """
    permute_op = PermuteOp(axes)
    return permute_op(ts, None)


def sigmoid(ts: Tensor) -> Tensor:
    exp_minus_x = exp(-ts)
    return 1 / (1 + exp_minus_x)


def cat(ts1: Tensor, ts2: Tensor, axes: int) -> Tensor:
    """
    Cat two tensors within given axes
    :param ts1:
    :param ts2:
    :param axes:
    :return:
    """
    cat_op = CatOp(axes)
    return cat_op(ts1, ts2)


def sum(ts: Tensor, axes: int) -> Tensor:
    sum_op = SumOp(axes)
    return sum_op(ts, None)


def binary_cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    """
    calculate mean cross entropy between input and target
    :param input: (batch_size, 1)
    :param target: (batch_size, 1)
    :return:
    """
    assert len(input.shape) == 2, 'binary cross entropy only used in 2 dim matrix'
    assert input.shape[1] == 1, 'binary shape[1] should be 1'
    loss = target * log(input) + (1 - target) * log(1 - input)
    return -sum(loss, 0) / input.shape[0]

