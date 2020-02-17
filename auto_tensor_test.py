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

        self.assertEqual(y1.data, np.array(3))
        self.assertEqual(y2.data, np.array(6))
        self.assertEqual(y3.data, np.array(7))

        y4.backward()
        self.assertEqual(x1.grad.data, np.array(2))
        self.assertEqual(x2.grad.data, np.array(1))

    def testMul(self):
        x1 = at.tensor([3, 4, 5], 'x1', requires_grad=True)
        x2 = at.tensor([2, 2, 2], 'x2', requires_grad=True)
        y1 = x1 * x2
        y1.backward()
        self.assertTrue(np.array_equal(y1.data, np.array([6, 8, 10])))
        self.assertTrue(np.array_equal(x1.grad.data, 2 * np.ones(3)))
        self.assertTrue(np.array_equal(x2.grad.data, np.array([3, 4, 5])))

        x1.zero_grad()
        x2.zero_grad()
        y1.zero_grad()
        y2 = y1 * x1 + x2
        y2.backward()
        self.assertTrue(np.array_equal(y2.data, x1.data ** 2 * x2.data + x2.data))
        self.assertTrue(np.array_equal(x1.grad.data, 2 * x1.data * x2.data))
        self.assertTrue(np.array_equal(x2.grad.data, x1.data ** 2 + np.ones(3)))


if __name__ == '__main__':
    unittest.main()
