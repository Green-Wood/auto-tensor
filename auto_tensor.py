from typing import Tuple

import numpy as np
from tensor import Tensor, tensor


def accumulate_grad(target: Tensor, grad: Tensor):
    """
    Accumulate gradient to target Tensor
    :param target:
    :param grad:
    :return: None
    """
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
        accumulate_grad(lhs, exp_op(lhs, None))


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
        if not axes:
            axes = (1, 0)
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


# singleton factory
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
add_op = AddOp()
mul_op = MulOp()
div_op = DivOp()
exp_op = ExpOp()
matmul = MatrixMulOp()


def exp(ts: Tensor) -> Tensor:
    """exp operation wrapper"""
    return exp_op(ts, None)


def view(ts: Tensor, new_shape: Tuple) -> Tensor:
    view_op = ViewOp(new_shape)
    return view_op(ts, None)


def transpose(ts: Tensor) -> Tensor:
    """equal to permute first 2 dim"""
    return permute(ts)


def permute(ts: Tensor, axes=None) -> Tensor:
    """
    Same as np.reshape
    :param ts: tensor to permute
    :param axes: if None equals to transpose permute(1, 0)
    :return:
    """
    permute_op = PermuteOp(axes)
    return permute_op(ts, None)
