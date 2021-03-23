import unittest

class TestOrbitalSimilarity(unittest.TestCase):

    def test_inertia_difference(self):
        from y4_python.python_modules.orbital_similarity import inertia_difference

        m1 = (1,1,1)
        m2 = (2,2,2)

        expected = 3/36

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)

        m1 = (1,1,1)
        m2 = (1,1,1)

        expected = 0

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)

        m1 = (1,1,1)
        m2 = (3,3,3)
 
        expected = 12/81

        result = inertia_difference(m1, m2)

        self.assertEqual(expected, result)