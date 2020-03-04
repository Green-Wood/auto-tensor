import unittest
import auto_tensor as at
import numpy as np
from auto_tensor import nn
import torch


class BasicFourOps(unittest.TestCase):
    def testAdd(self):
        x1 = at.tensor(1, 'x1', requires_optim=True)
        x2 = at.tensor(2, 'x2', requires_optim=True)
        y1 = x1 + x2
        y2 = x1 + 5
        y3 = 6 + x1
        y4 = y1 + x1

        self.assertEqual(np.array(3), y1.data)
        self.assertEqual(np.array(6), y2.data)
        self.assertEqual(np.array(7), y3.data)

        y4.backward()
        self.assertEqual(np.array(2), x1.grad)
        self.assertEqual(np.array(1), x2.grad)

    def testMul(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_optim=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_optim=True)
        y1 = x1 * x2
        y1.backward()
        self.assertTrue(np.array_equal(np.array([6, 8, 10]), y1.data))
        self.assertTrue(np.array_equal(2 * np.ones(3), x1.grad))
        self.assertTrue(np.array_equal(np.array([3, 4, 5]), x2.grad))

        x1.zero_grad()
        x2.zero_grad()
        y1.zero_grad()
        y2 = y1 * x1 + x2
        y2.backward()
        self.assertTrue(np.array_equal(x1.data ** 2 * x2.data + x2.data, y2.data))
        self.assertTrue(np.array_equal(2 * x1.data * x2.data, x1.grad))
        self.assertTrue(np.array_equal(x1.data ** 2 + np.ones(3), x2.grad))

    def testSub(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_optim=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_optim=True)
        y1 = x1 - x2
        y2 = x1 - 1
        y3 = 3 - x2
        y1.backward()

        self.assertTrue(np.array_equal(x1.data - x2.data, y1.data))
        self.assertTrue(np.array_equal(x1.data - 1, y2.data))
        self.assertTrue(np.array_equal(3 - x2.data, y3.data))
        self.assertTrue(np.array_equal(np.ones(3), x1.grad))
        self.assertTrue(np.array_equal(-np.ones(3), x2.grad))

    def testDiv(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_optim=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_optim=True)
        y1 = x1 / x2
        y2 = x2 / 2
        y1.backward()

        self.assertTrue(np.array_equal(x1.data / x2.data, y1.data))
        self.assertTrue(np.array_equal(y2.data, x2.data / 2))
        self.assertTrue(np.array_equal(1 / x2.data, x1.grad))
        self.assertTrue(np.array_equal(-x1.data / x2.data ** 2, x2.grad))

    def testSquare(self):
        x = at.tensor([2, 3, 4], 'x', requires_optim=True)
        y = x ** 5
        y.backward()

        self.assertTrue(np.array_equal(5 * x.data ** 4, x.grad))

    def testComplex1(self):
        x = at.tensor([1, 2, 3], 'x', requires_optim=True)
        y = at.tensor([2, 3, 4], 'y', requires_optim=True)
        z = x / (x + y)
        z.backward()

        expect_x_grad = y.data / (x.data + y.data) ** 2
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())

    def testComplex2(self):
        x = at.tensor([1, 2, 3], 'x', requires_optim=True)
        y = at.tensor([2, 3, 4], 'y', requires_optim=True)
        z = x ** 2 / (x + y)
        z.backward()

        expect_x_grad = (x.data ** 2 + 2 * x.data * y.data) / (x.data + y.data) ** 2
        print('\nexpect x grad: {}\ntrue x grad: {}'.format(expect_x_grad, x.grad))
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())

    def testComplex3(self):
        x = at.tensor([1, 2, 3], 'x', requires_optim=True)
        y = at.tensor([2, 3, 4], 'y', requires_optim=True)
        z = (x ** 2 + x - y) / (2 * x + y)

        expect_x_grad = ((2 * x.data + 1) * (2 * x.data + y.data) - 2 * (x.data ** 2 + x.data - y.data)) / (
                    2 * x.data + y.data) ** 2
        expect_y_grad = -(x.data ** 2 + 3 * x.data) / (2 * x.data + y.data) ** 2
        expect_z = (x.data ** 2 + x.data - y.data) / (2 * x.data + y.data)

        z.backward()
        print('\nexpect x grad: {}\ntrue x grad: {}'.format(expect_x_grad, x.grad))
        self.assertTrue(np.array_equal(expect_z, z.data))
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())
        self.assertTrue(np.isclose(expect_y_grad, y.grad).all())


class Complex(unittest.TestCase):
    def testExp(self):
        x = at.tensor([2, 3, 4], 'x', requires_optim=True)
        y = at.exp(x ** 2)
        y.backward()

        self.assertTrue(np.array_equal(np.exp(x.data ** 2), y.data))
        self.assertTrue(np.array_equal(2 * x.data * y.data, x.grad))

    def testLog(self):
        x = at.tensor([2, 3, 4], 'x', requires_optim=True)
        y = at.log(x ** 2)
        y.backward()

        self.assertTrue(np.array_equal(np.log(x.data ** 2), y.data))
        self.assertTrue(np.array_equal(2 / x.data, x.grad))

    def testView(self):
        x = at.tensor([2, 3, 4, 5], 'x', requires_optim=True)
        x1 = at.view(x, (2, 2))
        y = x1 ** 3
        y.backward()

        self.assertTrue(np.array_equal(x.data.reshape((2, 2)) ** 3, y.data))
        self.assertTrue(np.array_equal(3 * x.data ** 2, x.grad))

    def testTranspose(self):
        x = at.tensor([[2, 3, 4, 5]], 'x', requires_optim=True)
        x1 = at.transpose(x)
        y = x1 ** 3
        y.backward()

        self.assertTrue(np.array_equal(x.data.T ** 3, y.data))
        self.assertTrue(np.array_equal(3 * x.data ** 2, x.grad))

    def testMatMul(self):
        x1 = at.tensor([1, 2], 'x1', requires_optim=True)
        x2 = at.tensor([1, 2, 3, 4], 'x2', requires_optim=True)
        x1_view = at.view(x1, (2, 1))
        x2_view = at.view(x2, (2, 2))
        y = at.matmul(x2_view, x1_view)
        y.backward()

        expect_x1_grad = np.array([4, 6])
        expect_x2_grad = np.array([1, 2, 1, 2])
        self.assertTrue(np.array_equal(expect_x1_grad, x1.grad))
        self.assertTrue(np.array_equal(expect_x2_grad, x2.grad))

    def testSigmoid(self):
        x = at.tensor([2, 3, 4], 'x', requires_optim=True)
        u = at.sigmoid(x)
        y = 2 * u
        y.backward()

        expect_x_grad = u.data * (1 - u.data) * 2
        expect_y = (1 / (1 + np.exp(-x.data))) * 2
        self.assertTrue(np.isclose(expect_y, y.data).all())
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())

    def testCat(self):
        x1 = at.tensor([1, 2], 'x', requires_optim=True)
        x2 = at.tensor([[3, 3], [4, 5]], 'x', requires_optim=True)
        x1_view = at.view(x1, (2, 1))
        y = at.cat(x2, x1_view, axes=1)
        y = y ** 2
        y.backward()

        self.assertTrue(np.array_equal(2 * x1.data, x1.grad))
        self.assertTrue(np.array_equal(2 * x2.data, x2.grad))

    def testSum(self):
        x = at.tensor([[3, 2], [4, 5]], 'x', requires_optim=True)
        x_sum = at.sum(x, 1)
        y = x_sum ** 2
        y.backward()

        expect_x_grad = np.array([[10, 10], [18, 18]])
        self.assertTrue(np.array_equal(expect_x_grad, x.grad))

    def testRelu(self):
        x = at.tensor([[3, -2], [4, -5]], 'x', requires_optim=True)
        y = at.relu(x)
        y.backward()

        expect_y = np.array([[3, 0], [4, 0]])
        expect_grad = np.array([[1, 0], [1, 0]])
        self.assertTrue(np.array_equal(expect_y, y.data))
        self.assertTrue(np.array_equal(expect_grad, x.grad))


class NeuralNet(unittest.TestCase):

    def testLinear(self):
        x = at.tensor([[1, 2, 3], [3, 4, 5]], name='x', requires_optim=True)
        model = nn.Linear('linear', 3, 1, bias=False)
        y = model(x)
        y.backward()

        expect_w_grad = x.data.T @ np.ones((2, 1))
        self.assertTrue(np.array_equal(expect_w_grad, model.W.grad))

    def testBinaryCrossEntropy(self):
        x = at.tensor([0.1, 0.5, 0.8], name='x', requires_optim=True)
        y = at.tensor([0, 1, 1], name='y')
        x_view = at.view(x, (3, 1))
        y_view = at.view(y, (3, 1))
        loss = at.binary_cross_entropy(x_view, y_view)
        loss.backward()

        expect_loss = -np.mean(y.data * np.log(x.data) + (1 - y.data) * np.log(1 - x.data), keepdims=True)
        expect_x_grad = (y.data - x.data) / (x.shape[0] * x.data * (x.data - 1))
        self.assertTrue(np.array_equal(expect_loss, loss.data.flatten()))
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())

    def testSoftmax(self):
        x = at.tensor([[0.1, 0.5, 0.8]], name='x', requires_optim=True)
        y = at.softmax(x, 1)
        y.backward()

        expect_x_grad = y.data * (1 - y.data)
        self.assertAlmostEqual(1.0, np.sum(y.data))
        self.assertTrue(np.isclose(expect_x_grad, x.grad).all())

    def testCrossEntropy(self):
        at_a = at.tensor([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2], [0.2, 0.5, 0.1]], name='a')
        ts_a = torch.from_numpy(at_a.data)

        at_target = at.tensor([2, 0, 1], 'target')
        ts_target = torch.from_numpy(at_target.data)

        at_res = at.cross_entropy(at_a, at_target)
        ts_res = torch.nn.functional.cross_entropy(ts_a, ts_target)

        self.assertAlmostEqual(ts_res.item(), at_res.data[0][0])


if __name__ == '__main__':
    unittest.main()
