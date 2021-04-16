from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Iterator, List, Sequence, Set, Tuple, Union

from .orbital_calculations import MolecularOrbital, SerializedMolecularOrbital

import numpy as np  


def inertia_difference(moments1: Union[Sequence[float], Tuple[float, float, float]], moments2: Union[Sequence[float], Tuple[float, float, float]], ) -> float:
    """
    Calculate some difference between (calculated) two molecular orbitals
    based on their moment of inertia.

    New Distance: ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) / (x1*x2+y1*y2+z1*z2)

    Potential normalisation between 0->1:
        sq_distance between two vectors, A_ and B_ = |A_B_|^2
        from law of cosines:
            cos(x) = ( |A|^2 + |B|^2 - |A_B_|^2 ) / ( 2|A||B| )
            then squaring both sides gives
            cos^2(x) = ( |A|^2 + |B|^2 - |A_B_|^2 ) / ( 4 * |A|^2 * |B|^2 )

            cos^2(x) should vary between 0 and 1,
            0 when angle = pi/2, 
            1 when angle = 0

        However, consider we have our two moments: m1 = (1,2,3) m2 = (10,1000,10000)

    """
    m1 = sorted(moments1)
    m2 = sorted(moments2)
    length_squared = sum(
        [(m1[idx] - m2[idx])**2 for idx in range(len(moments1))]
    )
    # dot = sum([m1[idx] * m2[idx] for idx in range(len(moments1))])
    return length_squared**0.5


    # dot_squared = sum([m1[idx] * m2[idx] for idx in range(len(moments1))]) ** 2

    ### a . b = |a||b|cos(x)
    ### 1/cos(x) = |a||b| / (a.b)
    ### ( |a||b| / (a.b) )^2 = 1/cos^2(x) , which ranges between 0->1

def radial_distribution_difference(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital):
    """
    Radial Distribution is a vector of points, corresponding to f(r) for a specific range of r values.
      i.e. the vector [f(r) for r in np.arange(rmin,rmax,rstep)]

    This returns the norm (magnitude) of the difference of the two vectors.
    """

    dist1 = mo1["radial_distribution"]
    dist2 = mo2["radial_distribution"]

    return sum(
        ( dist1[idx] - dist2[idx]**2 for idx in range(len(dist1)) )
    ) ** 2



def IPR_difference(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital):
    """
    Calculate some difference between two (calculated) molecular orbitals
    based on there Inverse Participation Ratio.
    """
    return 0

def percent_heteroatom_difference(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital, atom_symbol: str):
    """
    Calculate some difference between two (calculated) molecular orbitals
    based on there percent weight on heteroatoms N, S and O.
    """
    symbol_key_map = {
        "N": "percent_on_N"
        , "O": "percent_on_O"
        , "S": "percent_on_S"
        , "P": "percent_on_P"
    }

    key = symbol_key_map[atom_symbol]
    diff = abs( mo1[key] - mo2[key] )
    
    return diff


def orbital_distance(
    mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital, inertia_coeff:float=1.
    , IPR_coeff:float=1, O_coeff:float=1
    , N_coeff:float=1, S_coeff:float=1, P_coeff:float=1
    , radial_distribution_coeff: float=1
    ):
    """
    Compute the Distance between 2 (calculated) molecular orbitals.

    This will likely be a combination of some distance of inertia, inverse part. ratio,
    and percent on heteroatoms: N and O.

    I suspect the most important will be O,N percent, then moment of inertia, then IPR.

    """
    if inertia_coeff == 0:
        inertia_diff = 0
    else:
        inertia_diff = inertia_coeff * inertia_difference(mo1["principal_moments"], mo2["principal_moments"])
    IPR_diff = IPR_difference(mo1, mo2)

    heteroatom_coeff_sum = O_coeff + N_coeff + S_coeff + P_coeff
    if heteroatom_coeff_sum == 0:
        heteroatom_diff = 0
    else:
        O_diff = percent_heteroatom_difference(mo1, mo2, "O")
        N_diff = percent_heteroatom_difference(mo1, mo2, "N")
        S_diff = percent_heteroatom_difference(mo1, mo2, "S")
        P_diff = percent_heteroatom_difference(mo1, mo2, "P")
        heteroatom_diff = ( O_coeff * O_diff + N_coeff * N_diff + S_diff * S_coeff + P_diff * P_coeff )  / heteroatom_coeff_sum


    distance = inertia_diff + IPR_coeff * IPR_diff + heteroatom_diff
    
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