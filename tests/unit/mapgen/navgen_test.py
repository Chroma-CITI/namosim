import unittest

from namosim.mapgen.mapgen import MapGen


class Test:
    def test_mapgen(self):
        x = MapGen(10, 10)
        x.gen_map()
        x.to_svg(60)
        assert True


if __name__ == "__main__":
    unittest.main()
