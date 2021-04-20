

from typing import List
from typing_extensions import TypedDict


import numpy as np

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from y4_python.python_modules.orbital_calculations import SerializedMolecularOrbital
from y4_python.python_modules.structural_similarity import structural_distance
from y4_python.python_modules.orbital_similarity import OrbitalDistanceKwargs, orbital_distance


class MetricParams(TypedDict):
    """
    This is good typing so that you can provide kwargs of 
    chemical distance metric as a dict (and easily store/write as json)
    then pass kwargs/params to chemical_distance(**params)
    or can be passed to KNN regressor as metric_params=params
    """
    fingerprint_list: List[ExplicitBitVect]
    homo_orbital_list: List[SerializedMolecularOrbital]
    lumo_orbital_list: List[SerializedMolecularOrbital]
    c_struct: float
    c_orbital: float
    inertia_coefficient: float
    IPR_coefficient: float
    N_coefficient: float
    O_coefficient: float
    S_coefficient: float
    P_coefficient: float
    radial_distribution_coeff: float

def chemical_distance(
    i: np.ndarray
    , j: np.ndarray
    , homo_coeff:float
    , lumo_coeff:float
    , fingerprint_list: List[ExplicitBitVect]
    , homo_orbital_list: List[SerializedMolecularOrbital]
    , lumo_orbital_list: List[SerializedMolecularOrbital]
    , c_orbital=0.0
    , c_struct=0.0
    , inertia_coefficient=0.0
    , IPR_coefficient=0.0
    , N_coefficient=0.0
    , O_coefficient=0.0
    , S_coefficient=0.0
    , P_coefficient=0.0
    , radial_distribution_coeff=0.0
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

    i_homo = homo_orbital_list[int(i[0])]
    j_homo = homo_orbital_list[int(j[0])]

    i_lumo = lumo_orbital_list[int(i[0])]
    j_lumo = lumo_orbital_list[int(j[0])]

    dist_orbital = 0
    if c_orbital:
        dist_orbital = c_orbital * orbital_distance(
            i_homo, i_lumo
            , j_homo, j_lumo
            , homo_coeff=homo_coeff
            , lumo_coeff=lumo_coeff
            , orbital_distance_kwargs=OrbitalDistanceKwargs(
                inertia_coeff=inertia_coefficient
                , IPR_coeff=IPR_coefficient
                , N_coeff=N_coefficient
                , O_coeff=O_coefficient
                , S_coeff=S_coefficient
                , P_coeff=P_coefficient
                , radial_distribution_coeff=radial_distribution_coeff
            )
        )

    dist_struct = 0
    if c_struct:
        dist_struct = c_struct * structural_distance(i_fp, j_fp)

    return dist_orbital + dist_struct