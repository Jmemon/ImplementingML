import unittest
from Modules.Linear import Linear
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
import numpy as np


xps = make_strategies_namespace(np)

class MyTestCase(unittest.TestCase):

    def __init__(self):
        self.parameters = np.ones((5, 4))
        self.layer = Linear(4, 5, parameters=self.parameters)

    @given(xps.arrays(dtype='float', shape=5))
    def test_forward(self, x):
        expected = np.matmul(self.parameters, x)
        self.assertEqual(expected, self.layer(x))


if __name__ == '__main__':
    unittest.main()
