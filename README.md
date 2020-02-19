# auto-tensor

![Python application](https://github.com/Green-Wood/auto-tensor/workflows/Python%20application/badge.svg)

> **Inspired by CSE-559W and PyTorch**

Auto-tensor is an (toy) auto differentiation tool that helps me dive into back propagation and deep learning system. By **constructing** **computation graph** and **defining some basic operation** such as `add, sub, mul, exp, log……`, we can build our own deep learning system out of scratch (Numpy).

- `auto_tensor.py`  define some useful helper functions and operations
- `nn.py` define neural network model and some common layers
- `optim.py` define optimizers
- `tensor.py` define main Tensor class using Numpy
- `auto_tensor_test.py` unittest file



## Examples

### 1. Logistic Regression

Import packages that we need.

Here, sklearn is only used to generate dataset and evaluate our model.

```python
import auto_tensor as at
from auto_tensor import nn
from auto_tensor import optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
```

Using some helper functions to help us define a logistic regression model

```python
class LogisticRegression(nn.Module):

    def __init__(self, feature_num):
        self.linear = nn.Linear('linear', in_dim=feature_num, out_dim=1, bias=True)

    def forward(self, x):
        u = self.linear(x)
        y = at.sigmoid(u)
        return y
```

Training

```python
epoch = 1000
lr = 0.1

model = LogisticRegression(X.shape[1])
optimizer = optim.SGD(model.params(), lr=lr)

for i in range(epoch):
    model.zero_grad()

    y_hat = model(X_train)
    # reshape y_train to (batch_size, 1)
    y_train_view = at.view(y_train, (y_train.shape[0], 1))  
    # calculate loss
    loss = at.binary_cross_entropy(y_hat, y_train_view)
    loss.backward()

    # take a step forward
    optimizer.step()

    if i % 100 == 0:
        y_predict_np = np.where(y_hat.data >= 0.5, 1, 0)
        train_acc = accuracy_score(y_train.data, y_predict_np)
        print('epoch: {}, loss: {}, train acc: {}'.format(i, loss, train_acc))
```

Finally, here is the final result for sklearn breast cancer dataset

```
epoch: 0, loss: tensor([[0.69314718]]), train acc: 0.6194225721784777
epoch: 100, loss: tensor([[0.10752918]]), train acc: 0.9816272965879265
epoch: 200, loss: tensor([[0.08930128]]), train acc: 0.9816272965879265
epoch: 300, loss: tensor([[0.08109473]]), train acc: 0.984251968503937
epoch: 400, loss: tensor([[0.07611094]]), train acc: 0.984251968503937
epoch: 500, loss: tensor([[0.07263933]]), train acc: 0.9868766404199475
epoch: 600, loss: tensor([[0.070022]]), train acc: 0.9868766404199475
epoch: 700, loss: tensor([[0.06794497]]), train acc: 0.9868766404199475
epoch: 800, loss: tensor([[0.06623663]]), train acc: 0.9868766404199475
epoch: 900, loss: tensor([[0.06479402]]), train acc: 0.9868766404199475

-------------start testing------------

test loss: tensor([[0.05554556]])
test acc: 0.9840425531914894
```

