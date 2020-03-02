import auto_tensor as at
from auto_tensor import nn
from auto_tensor import optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class LogisticRegression(nn.Module):

    def __init__(self, feature_num):
        self.linear = nn.Linear('linear', in_dim=feature_num, out_dim=1, bias=True)

    def forward(self, x):
        u = self.linear(x)
        y = at.sigmoid(u)
        return y


X, y = load_breast_cancer(True)
X = (X - np.mean(X, 0)) / np.std(X, 0)  # scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# convert numpy to Tensor
X_train = at.tensor(X_train, 'X_train', is_const=True)
X_test = at.tensor(X_test, 'X_test', is_const=True)
y_train = at.tensor(y_train, 'y_train', is_const=True)
y_test = at.tensor(y_test, 'y_test', is_const=True)

epoch = 1500
lr = 0.001

model = LogisticRegression(X.shape[1])
optimizer = optim.Adam(model.params(), lr=lr)

for i in range(epoch):
    model.zero_grad()

    y_hat = model(X_train)
    y_train_view = at.view(y_train, (y_train.shape[0], 1))  # reshape y_train to (batch_size, 1)
    loss = at.binary_cross_entropy(y_hat, y_train_view)
    loss.backward()

    # verify gradient
    expect_y_hat_grad = (y_train_view.data - y_hat.data) / (y_train_view.shape[0] * y_hat.data * (y_hat.data - 1))
    assert np.isclose(expect_y_hat_grad, y_hat.grad.data).all()

    # take a step forward
    optimizer.step()

    if i % 100 == 0:
        y_predict_np = np.where(y_hat.data >= 0.5, 1, 0)
        train_acc = accuracy_score(y_train.data, y_predict_np)
        print('epoch: {}, loss: {}, train acc: {}'.format(i, loss, train_acc))

print('\n-------------start testing------------\n')

y_hat = model(X_test)
y_test_view = at.view(y_test, (y_test.shape[0], 1))  # reshape y_test to (batch_size, 1)
loss = at.binary_cross_entropy(y_hat, y_test_view)
print('test loss: {}'.format(loss))

y_predict_np = np.where(y_hat.data >= 0.5, 1, 0)   # convert prob to two classes
acc = accuracy_score(y_test.data, y_predict_np)
print('test acc: {}'.format(acc))




