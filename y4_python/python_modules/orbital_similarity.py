from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Iterator, List, Sequence, Set, Tuple, Union

from .orbital_calculations import MolecularOrbital, SerializedMolecularOrbital

import numpy as np  


def inertia_difference(moments1: Union[Sequence[float], Tuple[float, float, float]], moments2: Union[Sequence[float], Tuple[float, float, float]], ) -> float:
    """
    Calculate some difference between (calculated) two molecular orbitals
    based on their moment of inertia.
    """

    return sum(
        [(moments1[idx] - moments2[idx])**2 for idx in range(len(moments1))]
    )


def IPR_difference(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital):
    """
    Calculate some difference between two (calculated) molecular orbitals
    based on there Inverse Participation Ratio.
    """
    return 0

def percent_heteroatom_difference(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital, atom_symbol: str):
    """
    Calculate some difference between two (calculated) molecular orbitals
    based on there percent weight on specified heteroatom.
    """
    return 0


def orbital_distance(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital, inertia_coeff:float=1., IPR_coeff:float=1, O_coeff:float=1, N_coeff:float=1):
    """
    Compute the Distance between 2 (calculated) molecular orbitals.

    This will likely be a combination of some distance of inertia, inverse part. ratio,
    and percent on heteroatoms: N and O.

    I suspect the most important will be O,N percent, then moment of inertia, then IPR.

    TODO: should the distance should be normalised to between 0 and 1?

    """

    inertia_diff = inertia_difference(mo1["principal_moments"], mo2["principal_moments"])
    IPR_diff = IPR_difference(mo1, mo2)
    O_diff = percent_heteroatom_difference(mo1, mo2, "O")
    N_diff = percent_heteroatom_difference(mo1, mo2, "N")

    distance = inertia_coeff * inertia_diff + IPR_coeff * IPR_diff + O_coeff * O_diff + N_coeff * N_diff
    
    return distance


def sort_molecular_orbital_pairs(
    orbitals: Union[Iterable[SerializedMolecularOrbital], Iterator[SerializedMolecularOrbital]]
    ) -> List[Tuple[SerializedMolecularOrbital, SerializedMolecularOrbital, float]]:
    """
    Given list of molecular orbitals, order them in pairs, from 
    most similar (least distant) to least similar (most distant).
    """
    from itertools import combinations

    pairs = combinations(orbitals, 2)

    similarities = [(x,y, orbital_distance(x,y)) for x,y in pairs]

    return sorted(similarities, key=lambda x: x[2])