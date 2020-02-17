import unittest
import auto_tensor as at
import numpy as np


class BasicFourOps(unittest.TestCase):
    def testAdd(self):
        x1 = at.tensor(1, 'x1', requires_grad=True)
        x2 = at.tensor(2, 'x2', requires_grad=True)
        y1 = x1 + x2
        y2 = x1 + 5
        y3 = 6 + x1
        y4 = y1 + x1

        self.assertEqual(np.array(3), y1.data)
        self.assertEqual(np.array(6), y2.data)
        self.assertEqual(np.array(7), y3.data)

        y4.backward()
        self.assertEqual(np.array(2), x1.grad.data)
        self.assertEqual(np.array(1), x2.grad.data)

    def testMul(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_grad=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_grad=True)
        y1 = x1 * x2
        y1.backward()
        self.assertTrue(np.array_equal(np.array([6, 8, 10]), y1.data))
        self.assertTrue(np.array_equal(2 * np.ones(3), x1.grad.data))
        self.assertTrue(np.array_equal(np.array([3, 4, 5]), x2.grad.data))

        x1.zero_grad()
        x2.zero_grad()
        y1.zero_grad()
        y2 = y1 * x1 + x2
        y2.backward()
        self.assertTrue(np.array_equal(x1.data ** 2 * x2.data + x2.data, y2.data))
        self.assertTrue(np.array_equal(2 * x1.data * x2.data, x1.grad.data))
        self.assertTrue(np.array_equal(x1.data ** 2 + np.ones(3), x2.grad.data))

    def testSub(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_grad=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_grad=True)
        y1 = x1 - x2
        y2 = x1 - 1
        y3 = 3 - x2
        y1.backward()

        self.assertTrue(np.array_equal(x1.data - x2.data, y1.data))
        self.assertTrue(np.array_equal(x1.data - 1, y2.data))
        self.assertTrue(np.array_equal(3 - x2.data, y3.data))
        self.assertTrue(np.array_equal(np.ones(3), x1.grad.data))
        self.assertTrue(np.array_equal(-np.ones(3), x2.grad.data))

    def testDiv(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_grad=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_grad=True)
        y1 = x1 / x2
        y2 = x2 / 2
        y1.backward()

        self.assertTrue(np.array_equal(x1.data / x2.data, y1.data))
        self.assertTrue(np.array_equal(y2.data, x2.data / 2))
        self.assertTrue(np.array_equal(1 / x2.data, x1.grad.data))
        self.assertTrue(np.array_equal(-x1.data / x2.data ** 2, x2.grad.data))

    def testSquare(self):
        x = at.tensor([2, 3, 4], 'x', requires_grad=True)
        y = x ** 5
        y.backward()

        self.assertTrue(np.array_equal(5 * x.data ** 4, x.grad.data))

    def testComplex1(self):
        x = at.tensor([1, 2, 3], 'x', requires_grad=True)
        y = at.tensor([2, 3, 4], 'y', requires_grad=True)
        z = x / (x + y)
        z.backward()

        expect_x_grad = y.data / (x.data + y.data) ** 2
        self.assertTrue(np.isclose(expect_x_grad, x.grad.data).all())

    def testComplex2(self):
        x = at.tensor([1, 2, 3], 'x', requires_grad=True)
        y = at.tensor([2, 3, 4], 'y', requires_grad=True)
        z = x ** 2 / (x + y)
        z.backward()

        expect_x_grad = (x.data**2 + 2*x.data*y.data) / (x.data+y.data) ** 2
        print('\nexpect x grad: {}\ntrue x grad: {}'.format(expect_x_grad, x.grad.data))
        self.assertTrue(np.isclose(expect_x_grad, x.grad.data).all())

    def testComplex3(self):
        x = at.tensor([1, 2, 3], 'x', requires_grad=True)
        y = at.tensor([2, 3, 4], 'y', requires_grad=True)
        z = (x ** 2 + x - y) / (2 * x + y)

        expect_x_grad = ((2*x.data+1) * (2*x.data+y.data) - 2 * (x.data ** 2 + x.data - y.data)) / (2*x.data+y.data)**2
        expect_y_grad = -(x.data**2+3*x.data) / (2*x.data + y.data) ** 2
        expect_z = (x.data ** 2 + x.data - y.data) / (2 * x.data + y.data)

        z.backward()
        print('\nexpect x grad: {}\ntrue x grad: {}'.format(expect_x_grad, x.grad.data))
        self.assertTrue(np.array_equal(expect_z, z.data))
        self.assertTrue(np.isclose(expect_x_grad, x.grad.data).all())
        self.assertTrue(np.isclose(expect_y_grad, y.grad.data).all())


class Complex(unittest.TestCase):
    def testExp(self):
        x = at.tensor([2, 3, 4], 'x', requires_grad=True)
        y = at.exp(x ** 2)
        y.backward()

        self.assertTrue(np.array_equal(np.exp(x.data**2), y.data))
        self.assertTrue(np.array_equal(2 * x.data * y.data, x.grad.data))

    def testView(self):
        x = at.tensor([2, 3, 4, 5], 'x', requires_grad=True)
        x1 = at.view(x, (2, 2))
        y = x1 ** 3
        y.backward()

        self.assertTrue(np.array_equal(x.data.reshape((2, 2)) ** 3, y.data))
        self.assertTrue(np.array_equal(3 * x.data ** 2, x.grad.data))

    def testTranspose(self):
        x = at.tensor([[2, 3, 4, 5]], 'x', requires_grad=True)
        x1 = at.transpose(x)
        y = x1 ** 3
        y.backward()

        self.assertTrue(np.array_equal(x.data.T ** 3, y.data))
        self.assertTrue(np.array_equal(3 * x.data ** 2, x.grad.data))

    def testMatMul(self):
        x1 = at.tensor([1, 2], 'x1', requires_grad=True)
        x2 = at.tensor([1, 2, 3, 4], 'x2', requires_grad=True)
        x1_view = at.view(x1, (2, 1))
        x2_view = at.view(x2, (2, 2))
        y = at.matmul(x2_view, x1_view)
        y.backward()

        expect_x1_grad = np.array([4, 6])
        expect_x2_grad = np.array([1, 2, 1, 2])
        self.assertTrue(np.array_equal(expect_x1_grad, x1.grad.data))
        self.assertTrue(np.array_equal(expect_x2_grad, x2.grad.data))


if __name__ == '__main__':
    unittest.main()
