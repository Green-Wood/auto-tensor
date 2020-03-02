from tensor import Tensor


class Module:

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def params(self):
        """
        attributes that need to compute gradient and optimize
        :return:
        """
        require_grad = [v for k, v in vars(self).items() if isinstance(v, Tensor) and v.requires_optim]
        for k, v in vars(self).items():
            if isinstance(v, Module):
                require_grad += v.params()
        return require_grad

    def forward(self, x):
        """
        Put an input x, get output of it
        :param x:
        :return:
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        clear all gradients
        :return:
        """
        ts_nn = [v for k, v in vars(self).items() if isinstance(v, Tensor) or isinstance(v, Module)]
        for item in ts_nn:
            # using duck type, both Tensor and Module and method zero_grad
            item.zero_grad()


class Linear(Module):

    def __init__(self, name: str, in_dim: int, out_dim: int, bias=False):
        """
        perform linear transformation
        :param in_dim:
        :param out_dim:
        :param bias: is add bias
        """
        import auto_tensor as at

        self.bias = bias
        self.name = name

        if not bias:
            self.W = at.zeros((in_dim, out_dim), '{}: W'.format(name), requires_optim=True, is_const=False)
        else:
            self.W = at.zeros((in_dim+1, out_dim), '{}: W'.format(name), requires_optim=True, is_const=False)

    def forward(self, x: Tensor):
        """
        Linear transformation
        :param x: (batch_size, feature_num)
        :return:
        """
        import auto_tensor as at

        assert len(x.shape) == 2, 'input x dim should be 2-dim'
        assert x.shape[1] == self.W.shape[0] or (self.bias and x.shape[1] + 1 == self.W.shape[0]), \
            'x dim[1]: {} should equals to in_dim: {}'.format(x.shape[1], self.W.shape[0])

        if self.bias:
            bias = at.ones((x.shape[0], 1), '{}: b'.format(self.name), requires_optim=False, is_const=True)
            x = at.cat(x, bias, 1)

        y = at.matmul(x, self.W)

        return y
