import unittest

class TestOrbitalSimilarity(unittest.TestCase):

    def test_inertia_difference(self):
        from y4_python.python_modules.orbital_similarity import inertia_difference

        m1 = (1,1,1)
        m2 = (2,2,2)

        expected = 3

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)

        m1 = (1,1,1)
        m2 = (1,1,1)

        expected = 0

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)

        m1 = (1,1,1)
        m2 = (-1,-1,-1)

        expected = 12

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)