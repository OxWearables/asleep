# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import importlib
import unittest

import numpy as np

from asleep.simple import add_one


class TestSimple(unittest.TestCase):

    def test_add_one(self) -> None:
        self.assertEqual(add_one(5), 6)

    def test_get_sleep_module_imports(self) -> None:
        self.assertGreaterEqual(
            np.lib.NumpyVersion(np.__version__),
            np.lib.NumpyVersion("1.24.0"),
        )
        importlib.import_module("asleep.get_sleep")


if __name__ == '__main__':
    unittest.main()
