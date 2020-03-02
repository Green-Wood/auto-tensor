from typing import Tuple

import numpy as np
from tensor import Tensor, tensor, ones, zeros
import nn
import optim


def accumulate_grad(target: Tensor, grad: np.ndarray):
    """
    Accumulate gradient to target Tensor
    :param target:
    :param grad:
    :return: None
    """
    # if this is a const, just return
    if target.is_const:
        return

    if isinstance(target.operation, SumOp):
        assert target.shape == grad.shape, \
            'Cannot take derivative of a sum up tensor explicitly.\n' \
            'Please avoid broadcast a sum up tensor and choose another way to calculate.\n' \
            'See why softmax as a basic operation'

    assert target.shape == grad.shape, \
        'tensor and gradient shape not compatible. Tensor: {}, Gradient: {}'.format(target.shape, grad.shape)

    target.grad += grad


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

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
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

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        accumulate_grad(lhs, acc_grad)
        accumulate_grad(rhs, acc_grad)


class OnesLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.ones_like(lhs.data)
        new_name = 'OnesLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        pass


class ZerosLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.zeros_like(lhs.data)
        new_name = 'ZerosLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        pass


class MulOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data * rhs.data
        new_name = '({}*{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        accumulate_grad(lhs, rhs.data * acc_grad)
        accumulate_grad(rhs, lhs.data * acc_grad)


class DivOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data / rhs.data
        new_name = '({}/{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        numerator_grad = np.ones_like(lhs.data) / rhs.data
        denominator_grad = (-lhs.data) / (rhs.data ** 2)
        accumulate_grad(lhs, numerator_grad * acc_grad)
        accumulate_grad(rhs, denominator_grad * acc_grad)


class ExpOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.exp(lhs.data)
        new_name = 'exp({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        assert not rhs
        this_grad = np.exp(lhs.data)
        accumulate_grad(lhs, this_grad * acc_grad)


class ViewOp(Operation):
    """Not a singleton, sacrifice performance to maintain interface consistency"""

    def __init__(self, new_shape: Tuple):
        self.new_shape = new_shape

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = lhs.data.reshape(self.new_shape)
        new_name = '(view({},{}))'.format(lhs.name, self.new_shape)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        assert not rhs
        acc_grad_reshape = acc_grad.reshape(lhs.shape)
        accumulate_grad(lhs, acc_grad_reshape)


class PermuteOp(Operation):

    def __init__(self, axes: Tuple):
        self.axes = axes

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.transpose(lhs.data, self.axes)
        new_name = '(permute({},{}))'.format(lhs.name, self.axes)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        assert not rhs
        new_axes = [0] * len(self.axes)
        # permute back to original space
        for k, v in enumerate(self.axes):
            new_axes[v] = k
        acc_grad_permute = np.transpose(acc_grad, tuple(new_axes))
        accumulate_grad(lhs, acc_grad_permute)


class MatrixMulOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = np.matmul(lhs.data, rhs.data)
        new_name = 'matmul({},{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        lhs_trans = np.transpose(lhs.data)
        rhs_trans = np.transpose(rhs.data)
        accumulate_grad(lhs, acc_grad @ rhs_trans)
        accumulate_grad(rhs, lhs_trans @ acc_grad)


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

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        lhs_len = lhs.shape[self.axes]
        rhs_len = rhs.shape[self.axes]
        lhs_grad, rhs_grad, _ = np.split(acc_grad, [lhs_len, lhs_len + rhs_len], self.axes)
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

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        repeat_len = lhs.shape[self.axes]
        acc_grad = np.repeat(acc_grad, repeat_len, axis=self.axes)
        accumulate_grad(lhs, acc_grad)


class LogOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.log(lhs.data)  # TODO log(0)
        new_name = 'log({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        assert not rhs
        curr_grad = 1.0 / lhs.data  # TODO overflow
        accumulate_grad(lhs, curr_grad * acc_grad)


class SoftmaxOp(Operation):

    def __init__(self, axes):
        self.axes = axes

    def cal_softmax_np(self, data: np.ndarray):
        number = np.max(data, axis=self.axes, keepdims=True)   # avoid overflow
        exp_input = np.exp(data - number)
        denominator = np.sum(exp_input, axis=self.axes, keepdims=True)
        new_data = exp_input / denominator
        return new_data

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = self.cal_softmax_np(lhs.data)
        new_name = 'softmax({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: np.ndarray):
        assert not rhs
        y = self.cal_softmax_np(lhs.data)
        accumulate_grad(lhs, y * (1 - y) * acc_grad)


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
    calculate mean binary cross entropy between input and target
    :param input: (batch_size, 1)
    :param target: (batch_size, 1)
    :return:
    """
    assert input.shape == target.shape, 'input and target have different shape!'
    assert len(input.shape) == 2, 'binary cross entropy only used in 2 dim matrix'
    assert input.shape[1] == 1, 'binary shape[1] should be 1'
    loss = target * log(input) + (1 - target) * log(1 - input)
    return -sum(loss, 0) / input.shape[0]


def softmax(input: Tensor, axes: int) -> Tensor:
    softmax_op = SoftmaxOp(axes)
    return softmax_op(input, None)


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    """
    calculate mean binary cross entropy between input and target
    :param input: (batch_size, classes)
    :param target: (batch_size, )   1-dim
    :return:
    """
    normalize = softmax(input, 1)
    norm_log = log(normalize)

    np_one_hot = np.eye(input.shape[1])[target.data]
    tensor_one_hot = tensor(np_one_hot, 'one-hot', False, True)

    mask = -norm_log * tensor_one_hot
    mask_sum = sum(mask, 1)
    loss = sum(mask_sum, 0)

    return loss / input.shape[0]



