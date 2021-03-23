
from collections import UserList
from itertools import combinations
import os
import bisect
from typing import Any, Iterable, List, Tuple

import sklearn
from y4_python.python_modules.orbital_similarity import orbital_distance

def my_insort_left(a, x, lo=0, hi=None):
    """
    Insort left modified to use lambda x: x[0] for comparisons
    """

    x_key = x[0]


    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid][0] < x_key: lo = mid+1
        else: hi = mid
    a.insert(lo, x)


class MyList(UserList):
    """
    A list but we keep it in ascending order,
    also has compare_insert function. 

    """

    def __init__(self, highest:bool, key=None, *args, **kwargs):
        "if highest==True then this list should keep the highest values vice versa"
        super().__init__(*args, **kwargs)
        self.highest = highest
        if key == None:
            self.key = lambda x: x
        else:
            self.key = key

    # def __setitem__(self, *args, **kwargs):
    #     super().__setitem__(*args, **kwargs)
    #     self.data.sort()

    # def append(self, *args, **kwargs):
    #     super().append(*args, **kwargs)
    #     self.data.sort()

    def compare_replace(self, item):
        """
        Compare item with the lowest/highest in self.items
        If lowest == True, compare with lowest, else compare with highest.

        If the item is higher than the lowest item then it should be in here, 
        so we replace the item with the lowest current item and return the item 
        that was popped. (To be fed into compare with another list).
        """
        key_item = self.key(item)
        if self.highest: # this is highest list
            if key_item > self.key(self.data[0]):
                # replace the lowest
                popped = self.pop(0)
                my_insort_left(self, item)
                return popped
            else:
                return False
        else: # this is lowest list
            if key_item < self.key(self.data[-1]):
                # replace the highest
                popped = self.pop(-1)
                my_insort_left(self, item)
                # bisect.insort(self, item, key=)
                return popped
            else:
                return False


def algo(l: Iterable[Any], k, key=lambda x: x) -> Tuple[List[Any], List[Any]]:
    """
    Given a list of items, get the k most and least values from the list. 
    If items themselves aren't comparable with < and > operators, then a key
    should be supplied such that this becomes the case
    eg if item is Tuple[int, str] then key should be lambda x: x[0] to compare the int.

    Can assume k will always be smaller than len(list) / 2
    There can be duplicates.

    Most importantly, if l is an iterable (such as map or generator) then this algorithm
    will use basically zero memory.

    The my_left_insort operation is probably the most time_consuming as O(n)


    """

    highest = MyList(highest=True, key=key)
    lowest = MyList(highest=False, key=key)

    map_ = {True: highest, False:lowest}
    for item in l:
        if len(highest) < k:
            my_insort_left(highest, item)
        else:
            popped = highest.compare_replace(item)
            if len(lowest) < k:
                if popped:
                    my_insort_left(lowest, popped)
                else:
                    my_insort_left(lowest, item)
            else:
                h_or_l = not popped # when True check is highest, and when False check is lowest
                if not popped:
                    popped = lowest.compare_replace(item)                    
                while popped:
                    check = map_[h_or_l]
                    popped = check.compare_replace(popped)                    
                    h_or_l = not h_or_l # change check from highest -> lowest or vice versa

    return list(highest), list(lowest)


def not_fast(l: List[int], k, key=lambda x: x) -> Tuple[List[int], List[int]]:
    "This will work, but its not as fast, obviously?"
    l = l.copy()
    least = []
    most = []
    for _ in range(k):
        least.append(
            l.pop(
                l.index(min(l, key=key))
            )
        )
    for _ in range(k):
        most.append(
            l.pop(
                l.index(max(l, key=key))
            )
        )
    return most, least

from time import perf_counter
# l = [1,324,6,23,346,754,324,3546,8546,57,32,34]
# k = 2
# print(not_fast(l, k))
# print(algo(l,k))
# print(algo(l,k) == not_fast(l,k))

# from random import randint
# l = [randint(1,1000) for x in range(1_000)]
# k = 100
# t0 = perf_counter()
# nf = not_fast(l,k)
# t1 = perf_counter()
# print(f"not_fast took {t1 - t0} seconds")
# t0 = perf_counter()
# nf = algo(l,k)
# t1 = perf_counter()
# print(f"algo took {t1 - t0} seconds")


def sklearnNeighbours(n_neighbors=2):
    """
    Minimum example for sklearn.NearestNeighbors.kneighbors

    ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors
    """
    from sklearn.neighbors import NearestNeighbors
    samples = [
        [0., 0., 0.]
        , [1., 1., 1.]
        , [2., 2., 2.]
    ]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(samples)

    kneigh = neigh.kneighbors([
            [1.5,1.5,1.5]
            , [0.5,0.5,0.5]
        ]
    ) # returns (array(distances), array(indices))
    print(kneigh[1])
    print(kneigh)


if __name__ == "__main__":
    from .python_modules.database import DB 

    # db = DB(os.path.join("y4_python","molecule_database-eV.db"))

    # all_ = db.get_all()[:10_000]

    # pairs = combinations(all_, 2)

    # pairs_map = map(
    #     lambda pair: (orbital_distance(
    #         pair[0][5]
    #         , pair[1][5]
    #     ),) + pair
    #     , pairs
    # )
    # k=10
    # t0 = perf_counter()
    # most, least = algo(
    #     pairs_map
    #     , k=k
    #     , key=lambda x: x[0]
    # )
    # t1 = perf_counter()
    # # print(most, least)
    # print(f"took {t1-t0} seconds.")

    # pairs = combinations(all_, 2)

    # t0 = perf_counter()
    # distances = []
    # for x,y in pairs:
    #     i = x[5]
    #     j = y[5]
    #     distance = orbital_distance(i,j)
    #     distances.append(
    #         (
    #             distance
    #             , x # (molName, Epm7, Eblyp, smiles, fingerprints, serialized_mol_orb)
    #             , y # (molName, Epm7, Eblyp, smiles, fingerprints, serialized_mol_orb)
    #         )
    #     )

    # sortedDistances = sorted(distances, key=lambda x: x[0], reverse=True)
    # most, least = sortedDistances[-k:], sortedDistances[:k]
    # t1 = perf_counter()
    # #print(most, least)
    # print(f"took {t1-t0} seconds")
    sklearnNeighbours()

