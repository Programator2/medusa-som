import unittest

import som

class TestNeighborhood(unittest.TestCase):
    def setUp(self):
        self.som = som.SOM([])
        self.som.col_sz = 4
        self.som.row_sz = 4

    def test_edge(self):
        s = self.som
        neighborhood = s._topological_neighborhood(8, 1)
        expected_neighborhood = [4, 5, 9, 12, 13]
        self.assertCountEqual(neighborhood, expected_neighborhood)

    def test_corner(self):
        s = self.som
        neighborhood = s._topological_neighborhood(15, 1)
        expected_neighborhood = [10, 11, 14]
        self.assertCountEqual(neighborhood, expected_neighborhood)

    def test_inside(self):
        s = self.som
        neighborhood = s._topological_neighborhood(6, 1)
        expected_neighborhood = [1, 2, 3, 5, 7, 9, 10, 11]
        self.assertCountEqual(neighborhood, expected_neighborhood)


if __name__ == '__main__':
    unittest.main()
