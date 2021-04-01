import unittest

class TestSklearnNN(unittest.TestCase):

    def test_kneighbours(self):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        data = np.array([
            [0, 0]
            , [1, 1]
            , [2, 2]
            , [3, 3]
            , [4, 4]
            , [5, 5]
        ])

        # check itself isn't a nearest neighbor.

        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(data)

        distances, indices = neigh.kneighbors([[1,1],[4,4]])
        print(distances, indices)


if __name__ == "__main__":
    unittest.main()