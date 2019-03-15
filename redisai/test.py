import six
import json
import unittest
from unittest import TestCase
from .client import Client, Type

rai = None
port = 6379

class RedisAITestCase(TestCase):

    def setUp(self):
        global rai
        rai = Client(port=port)
        rai.flushdb()

    def testTensorSet(self):
        "Test basic TENSORSET"

        self.assertTrue(rai.tensorset("key", Type.FLOAT, ["1"], ["2"]))



if __name__ == '__main__':
    unittest.main()