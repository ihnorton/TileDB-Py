import unittest

import tiledb
from tiledb import TileDBError, readquery

ctx = tiledb.default_ctx()
a = tiledb.open("/tmp/axxxa1")


class BasicTest(unittest.TestCase):
    def __init__(self, r):
        self.r = r

    def test(self):
        try:
            self.r.test_err("foobar")
        except Exception as exc:
            assert isinstance(exc, tiledb.TileDBError)
            assert exc.message == "foobar"

        with self.assertRaises(TileDBError):
            self.r.set_ranges([[(3, "a")]])

        with self.assertRaises(TileDBError):
            self.r.set_ranges([[(0, 3)]])

        with self.assertRaisesRegex(
            TileDBError,
            "Failed to cast dim range '\(1.2344, 5.6789\)' to dim type UINT64.*$",
        ):
            self.r.set_ranges([[(1.2344, 5.6789)]])

        with self.assertRaisesRegex(
            TileDBError,
            "Failed to cast dim range '\('aa', 'bbbb'\)' to dim type UINT64.*$",
        ):
            self.r.set_ranges([[("aa", "bbbb")]])

        print("done")


r2 = readquery.ReadQuery(ctx, a, (1,), False)
BasicTest(r2).test()
