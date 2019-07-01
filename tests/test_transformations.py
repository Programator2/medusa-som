import unittest

import paths

class TestTransformations(unittest.TestCase):
    def test_same(self):
        original = '/home/user/test.txt'
        new = '/home/user/test.txt'
        ret = paths.find_transformation(original, new)
        self.assertEqual(ret, '')

    def test_up_prefix(self):
        original = '/home/user/typography/doc.pdf'
        new = '/home/user'
        ret = paths.find_transformation(original, new)
        self.assertEqual(ret, '..')

    def test_up_different(self):
        original = '/home/user/typography/doc.pdf'
        new = '/home/user/test.txt'
        ret = paths.find_transformation(original, new)
        self.assertEqual(ret, '..')

    def test_down_folder(self):
        new = '/home/user/typography/doc.pdf'
        original = '/home/user'
        ret = paths.find_transformation(original, new)
        self.assertEqual(ret, 'typography')

    def test_down_file(self):
        new = '/home/user/doc.pdf'
        original = '/home/user'
        ret = paths.find_transformation(original, new)
        self.assertEqual(ret, 'doc.pdf')

if __name__ == '__main__':
    unittest.main()
