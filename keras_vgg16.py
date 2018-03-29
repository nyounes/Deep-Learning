from keras.applications.vgg16 import VGG16
import unittest

model = VGG16(weights='imagenet', include_top=True)


class testTest(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
