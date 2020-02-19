from typing import Tuple, List

import numpy as np


class Tensor:
    # ！！！！！！！！！！！！！！！THIS IS A COMPUTE GRAPH, NOT A BINARY TREE！！！！！！！！！！！！！！！！
    def __init__(self,
                 data: np.ndarray,
                 name: str,
                 requires_grad: bool = False,
                 lhs=None,
                 rhs=None,
                 operation=None,
                 is_const=False):
        from auto_tensor import Operation

        self.data = data
        self.name = name
        self.requires_grad = requires_grad
        self.lhs: Tensor = lhs
        self.rhs: Tensor = rhs
        self.operation: Operation = operation
        self.grad: Tensor = None
        self.is_const = is_const

        self.shape: Tuple = self.data.shape

    def backward(self):
        """start backpropagation from current tensor, accumulate to each tensor's gradient"""
        from auto_tensor import ones_like

        def reversed_topo_sort() -> List[Tensor]:
            """Given a list of nodes, return a topological sort list of nodes ending in them."""
            visited = set()
            topo_order = []
            topo_sort_dfs(self, visited, topo_order)
            return reversed(topo_order)

        def topo_sort_dfs(ts: Tensor, visited, topo_order):
            """Post-order DFS"""
            if ts in visited or not ts or not ts.operation:
                return
            visited.add(ts)
            topo_sort_dfs(ts.lhs, visited, topo_order)
            topo_sort_dfs(ts.rhs, visited, topo_order)
            topo_order.append(ts)

        self.grad = ones_like(self, None)

        for t in reversed_topo_sort():
            t.operation.backward(t.lhs, t.rhs, t.grad)

    def zero_grad(self):
        """clear gradient"""
        self.grad = None

    def __add__(self, other):
        from auto_tensor import add_op
        other = check_tensor(other)
        return add_op(self, other)

    def __mul__(self, other):
        from auto_tensor import mul_op
        other = check_tensor(other)
        return mul_op(self, other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = check_tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = check_tensor(other)
        return other - self

    def __truediv__(self, other):
        from auto_tensor import div_op
        other = check_tensor(other)
        return div_op(self, other)

    def __rtruediv__(self, other):
        other = check_tensor(other)
        return other / self

    def __pow__(self, power, modulo=None):
        from auto_tensor import ones_like
        assert isinstance(power, int)

        res = ones_like(self, None)  # 1
        for i in range(power):
            res *= self
        return res

    def __str__(self):
        return 'tensor({})'.format(self.data)

    __radd__ = __add__
    __rmul__ = __mul__


def tensor(data, name: str, requires_grad: bool = False, is_const=False) -> Tensor:
    """Create Tensor user friendly"""
    if isinstance(data, np.ndarray):
        return Tensor(data, name, requires_grad)
    return Tensor(np.array(data), name, requires_grad=requires_grad, is_const=is_const)


def ones(shape: Tuple, name: str, requires_grad: bool = False, is_const=False) -> Tensor:
    """create all ones tensor"""
    data = np.ones(shape)
    return Tensor(data, name, requires_grad=requires_grad, is_const=is_const)


def zeros(shape: Tuple, name: str, requires_grad: bool = False, is_const=False) -> Tensor:
    """create all zeros tensor"""
    data = np.zeros(shape)
    return Tensor(data, name, requires_grad=requires_grad, is_const=is_const)


def check_tensor(data) -> Tensor:
    """check whether it is a scala, List or Tensor"""
    if not isinstance(data, Tensor):
        data = tensor(data, str(data), is_const=True)
    return data
