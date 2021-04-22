from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Iterator, List, Sequence, Set, Tuple, Union

from typing_extensions import TypedDict

from .orbital_calculations import MolecularOrbital, SerializedMolecularOrbital

import numpy as np  


def inertia_difference(
    moments1: Union[Sequence[float], Tuple[float, float, float]]
    , moments2: Union[Sequence[float], Tuple[float, float, float]], 
    ) -> float:
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

def radial_distribution_difference(RDF1: List[float], RDF2: List[float]):
    """
    Radial Distribution is a vector of points, corresponding to f(r) for a specific range of r values.
      i.e. the vector [f(r) for r in np.arange(rmin,rmax,rstep)]

    This returns the norm (magnitude) of the difference of the two vectors.
    """

    return sum(
        ( abs(RDF1[idx] - RDF2[idx]) for idx in range(len(RDF1)) )
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


class OrbitalDistanceKwargs(TypedDict):
    inertia_coeff:float
    IPR_coeff:float
    O_coeff:float
    N_coeff:float
    S_coeff:float
    P_coeff:float
    radial_distribution_coeff: float

def _mo_distance(mo1: SerializedMolecularOrbital, mo2: SerializedMolecularOrbital
    , inertia_coeff:float=0.0
    , IPR_coeff:float=0.0
    , O_coeff:float=0.0
    , N_coeff:float=0.0
    , S_coeff:float=0.0
    , P_coeff:float=0.0
    , radial_distribution_coeff: float=0.0
    ):
    """
    Compute the Distance between 2 (calculated) molecular orbitals.

    This will likely be a combination of some distance of inertia, inverse part. ratio,
    and percent on heteroatoms: N and O.

    I suspect the most important will be O,N percent, then moment of inertia.

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
        O_diff = percent_heteroatom_difference(mo1, mo2, "O") if O_coeff > 0 else 0 # avoid unnecessary computation
        N_diff = percent_heteroatom_difference(mo1, mo2, "N") if N_coeff > 0 else 0
        S_diff = percent_heteroatom_difference(mo1, mo2, "S") if S_coeff > 0 else 0
        P_diff = percent_heteroatom_difference(mo1, mo2, "P") if P_coeff > 0 else 0
        heteroatom_diff = ( O_coeff * O_diff + N_coeff * N_diff + S_diff * S_coeff + P_diff * P_coeff )  / heteroatom_coeff_sum

    if radial_distribution_coeff == 0:
        radial_distribution_diff = 0
    else:
        radial_distribution_diff = radial_distribution_coeff * radial_distribution_difference(mo1["radial_distribution"], mo2["radial_distribution"])

    distance = inertia_diff + heteroatom_diff + radial_distribution_diff
    
    return distance
    

def orbital_distance(
    homo1: SerializedMolecularOrbital, lumo1: SerializedMolecularOrbital
    , homo2: SerializedMolecularOrbital, lumo2: SerializedMolecularOrbital
    , homo_coeff
    , lumo_coeff
    , orbital_distance_kwargs: OrbitalDistanceKwargs
    ):
    if homo_coeff == 0: # avoid calculation time
        homo_dist = 0
    else:
        homo_dist = _mo_distance(homo1, homo2, **orbital_distance_kwargs)
    if lumo_coeff == 0:
        lumo_dist = 0
    else:
        lumo_dist = _mo_distance(lumo1, lumo2, **orbital_distance_kwargs)

    return ( homo_dist * homo_coeff + lumo_dist * lumo_coeff ) / ( homo_coeff + lumo_coeff )


def sort_molecular_orbital_pairs(
    orbitals: Union[Iterable[SerializedMolecularOrbital], Iterator[SerializedMolecularOrbital]]
    , orbital_distance_kwargs: dict = {}
    ) -> List[Tuple[SerializedMolecularOrbital, SerializedMolecularOrbital, float]]:
    """
    Given list of molecular orbitals, order them in pairs, from 
    most similar (least distant) to least similar (most distant).
    """
    from itertools import combinations

    pairs = combinations(orbitals, 2)

    similarities = [(x,y, orbital_distance(x,y, **orbital_distance_kwargs)) for x,y in pairs]

    return sorted(similarities, key=lambda x: x[2])