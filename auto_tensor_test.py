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

        self.assertEquals(y1.data, np.array(3))
        self.assertEquals(y2.data, np.array(6))
        self.assertEquals(y3.data, np.array(7))

        y1.backward()
        self.assertEquals(x1.grad.data, np.array(1))
        self.assertEquals(x2.grad.data, np.array(1))


if __name__ == '__main__':
    unittest.main()
