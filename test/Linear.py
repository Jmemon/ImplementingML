import unittest
from Modules.Linear import Linear
from hypothesis import given


class MyTestCase(unittest.TestCase):

    @given()
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
