import unittest
from y4_python.algorithm_testing import algo


class TestAlgorithmTesting(unittest.TestCase):

    def test_algo(self):
        l = (
            12
            , 14
            , 14
            , 18
            , 2
            , 1
            , 19
            , 25
            , 52
            ,723
            ,7524
            ,7896
            ,3524
            ,7896
            ,2
            ,7546
            ,312
            ,4
            ,6
            ,34
            ,68
        )
        map_l = map(
            lambda x: (x, str(x))
            , l
        )
        k=5
        most, least = algo(map_l, k=k, key=lambda x: x[0])
        copy = list(l)
        expectedMost = [
            (x, str(x)) for x in sorted([copy.pop(copy.index(max(copy))) for _ in range(k)])
        ]
        expectedLeast = [
            (x, str(x)) for x in sorted([copy.pop(copy.index(min(copy))) for _ in range(k)])
        ]
        self.assertEqual(most, expectedMost)
        self.assertEqual(least, expectedLeast)