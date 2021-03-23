

from typing import List
from typing_extensions import TypedDict


import numpy as np

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from y4_python.python_modules.orbital_calculations import SerializedMolecularOrbital
from y4_python.python_modules.structural_similarity import structural_distance
from y4_python.python_modules.orbital_similarity import orbital_distance


class MetricParams(TypedDict):
    fingerprint_list: List[ExplicitBitVect]
    molecular_orbital_list: List[SerializedMolecularOrbital]
    c_orbital: float
    inertia_coefficient: float
    IPR_coefficient: float
    N_coefficient: float
    O_coefficient: float
    S_coefficient: float
    P_coefficient: float
    c_struct: float

def chemical_distance(
    i: np.ndarray
    , j: np.ndarray
    , fingerprint_list: List[ExplicitBitVect]
    , molecular_orbital_list: List[SerializedMolecularOrbital]
    , c_orbital=1.0
    , c_struct=1.0
    , inertia_coefficient=1.0
    , IPR_coefficient=1.0
    , N_coefficient=1.0
    , O_coefficient=1.0
    , S_coefficient=1.0
    , P_coefficient=1.0
    ):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    A function which takes two one-dimensional numpy arrays, and returns a distance. 
    
    Returns Distance between molecule i and j (D_ij) such that:
    D_ij = 1 - T(i,j)  where T(i,j) is the tanimoto similarity between molecule i and j.

    Note that in order to be used within the BallTree, the distance must be a true metric. 
    i.e. it must satisfy the following properties:
    Non-negativity: d(x, y) >= 0
    Identity: d(x, y) = 0 if and only if x == y
    Symmetry: d(x, y) = d(y, x)
    Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
    

    Our current combination for multiple distance functions is as follows:

    D(i,j) = c_inertia * inertia_distance(i,j) + c_struct * structural_distance(i,j)

    """
    i_fp= fingerprint_list[int(i[0])]
    j_fp = fingerprint_list[int(j[0])]

    i_mo = molecular_orbital_list[int(i[0])]
    j_mo = molecular_orbital_list[int(j[0])]

    dist_orbital = 0
    if c_orbital:
        dist_orbital = c_orbital * orbital_distance(
            i_mo, j_mo
            , inertia_coeff=inertia_coefficient
            , IPR_coeff=IPR_coefficient
            , N_coeff=N_coefficient
            , O_coeff=O_coefficient
            , S_coeff=S_coefficient
            , P_coeff=P_coefficient
        )

    dist_struct = 0    
    if c_struct:
        dist_struct = c_struct * structural_distance(i_fp, j_fp)

    return dist_orbital + dist_struct