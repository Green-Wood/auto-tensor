import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import auto_tensor as at
from auto_tensor import nn
from auto_tensor import optim
from sklearn.metrics import accuracy_score
import numpy as np


class Model(nn.Module):

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.linear = nn.Linear('linear', in_dim=in_features, out_dim=out_features, bias=False)

    def forward(self, x):
        # flatten
        x = at.view(x, (x.shape[0], self.in_features))
        y = self.linear(x)
        return y


epoch = 5
lr = 0.01
batch_size = 64

transform = transforms.Compose([
    # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.0)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transforms.ToTensor())

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


model = Model(in_features=28 * 28, out_features=10)
optimizer = optim.Adam(model.params(), lr=lr)

for i in range(epoch):
    for j, (images, labels) in enumerate(trainloader):
        model.zero_grad()

        images = at.tensor(images.numpy(), 'train image')
        labels = at.tensor(labels.numpy(), 'labels')

        y_hat = model(images)
        loss = at.cross_entropy(y_hat, labels)
        loss.backward()

        # take a step forward
        optimizer.step()

        if j % 100 == 0:
            y_predict_np = np.argmax(y_hat.data, axis=1)
            train_acc = accuracy_score(labels.data, y_predict_np)
            print('epoch: {}, loss: {}, train acc: {}'.format(i, loss, train_acc))

print('\n-------------start testing------------\n')


test_loss = 0
test_acc = 0
for images, labels in testloader:
    images = at.tensor(images.numpy(), 'train image')
    labels = at.tensor(labels.numpy(), 'labels')

    y_hat = model(images)
    loss = at.cross_entropy(y_hat, labels)
    test_loss += loss.data

    y_predict_np = np.argmax(y_hat.data, axis=1)
    acc = accuracy_score(labels.data, y_predict_np)
    test_acc += acc

test_loss = test_loss * batch_size / len(testset)
test_acc = test_acc * batch_size / len(testset)

print('test loss: {}'.format(test_loss))
print('test acc: {}'.format(test_acc))

