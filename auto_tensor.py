import numpy as np
from tensor import Tensor, tensor


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

    def accumulate_grad(self, target: Tensor, grad: Tensor):
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


class AddOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data + rhs.data
        new_name = '({}+{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        self.accumulate_grad(lhs, acc_grad)
        self.accumulate_grad(rhs, acc_grad)


class OnesLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.ones_like(lhs.data)
        new_name = 'OnesLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        lhs.grad = zeros_like(lhs, None)


class ZerosLikeOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        assert not rhs
        new_data = np.zeros_like(lhs.data)
        new_name = 'ZerosLike({})'.format(lhs.name)
        return Tensor(new_data, new_name, lhs=lhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        assert not rhs
        lhs.grad = zeros_like(lhs, None)


class MulOp(Operation):

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        new_data = lhs.data * rhs.data
        new_name = '({}*{})'.format(lhs.name, rhs.name)
        return Tensor(new_data, new_name, lhs=lhs, rhs=rhs, operation=self)

    def backward(self, lhs: Tensor, rhs: Tensor, acc_grad: Tensor):
        self.accumulate_grad(lhs, rhs * acc_grad)
        self.accumulate_grad(rhs, lhs * acc_grad)


# singleton factory
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
add = AddOp()
mul = MulOp()
