import unittest

import som

class TestTopologicalDistance(unittest.TestCase):
    def setUp(self):
        self.som = som.SOM([])
        self.som.col_sz = 7
        self.som.row_sz = 7

    def test_1(self):
        s = self.som
        distance = s._topological_distance(23, 31)
        self.assertEqual(distance, 1)

    def test_2(self):
        s = self.som
        distance = s._topological_distance(23, 10)
        self.assertEqual(distance, 2)

    def test_6(self):
        s = self.som
        distance = s._topological_distance(42, 3)
        self.assertEqual(distance, 6)


if __name__ == '__main__':
    unittest.main()
