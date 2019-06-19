import unittest

from som import SOM

class TestDistance(unittest.TestCase):
    def test_same(self):
        original = '/home/user/test.txt'
        new = '/home/user/test.txt'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 0)

    def test_up_prefix(self):
        original = '/home/user/typography/doc.pdf'
        new = '/home/user'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 2)

    def test_up_different(self):
        original = '/home/user/typography/doc.pdf'
        new = '/home/user/test.txt'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 3)

    def test_down_folder(self):
        new = '/home/user/typography/doc.pdf'
        original = '/home/user'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 2)

    def test_down_file(self):
        new = '/home/user/doc.pdf'
        original = '/home/user'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 1)

    def test_across_root(self):
        new = '/home/user/firefox/bookmarks'
        original = '/var/log/gnome'
        distance = SOM.path_distance(original, new)
        self.assertEqual(distance, 7)

if __name__ == '__main__':
    unittest.main()
