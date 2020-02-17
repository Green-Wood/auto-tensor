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
        from auto_tensor import Operation

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
        from auto_tensor import ones_like

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

    def zero_grad(self):
        """clear gradient"""
        self.grad = None

    def __add__(self, other):
        from auto_tensor import add
        other = check_tensor(other)
        return add(self, other)

    def __mul__(self, other):
        from auto_tensor import mul
        other = check_tensor(other)
        return mul(self, other)

    __radd__ = __add__
    __rmul__ = __mul__


def tensor(data, name: str, requires_grad: bool = False) -> Tensor:
    """Create Tensor user friendly"""
    if isinstance(data, np.ndarray):
        return Tensor(data, name, requires_grad)
    return Tensor(np.array(data), name, requires_grad)


def check_tensor(data) -> Tensor:
    """check whether it is a scala, List or Tensor"""
    if not isinstance(data, Tensor):
        data = tensor(data, str(data))
    return data
