"""
Possible descriptors that consider the nature and shape of orbitals

For a given orbital, it is possible to associate the weight of that orbital on each atom.
For semiempirical methods, the weight on each atom is the sum of the square of the coefficient
for all atomic orbital centred on that atom.
The sum of the weights on each atom should be 1.

To each atom i we therefore associate the atomic symbol S_i the cartesian coordinates x_i, y_i, z_i,
and the weight of the orbital WH_i. 
These can be used to define descriptors with the property that molecules with similar descriptors have similar orbitals.



The overall "shape" of the orbital can be described by the orbital moments of inertia. 
This is computed with 15 lines of code from the coordinates and the orbital weight essentially using the orbital weight
instead of the mass of the atom. The formulas are here: https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor


contents of the script:
    DONE - calculate weight on the atom
    TODO - calc IPR for orbital
    TODO - calc. weight on given heteroatom
    TODO - calc inertia tensor for orbital
"""

from typing import Dict, List, NamedTuple, Tuple
import json

from typing_extensions import TypedDict

import numpy as np

class AtomicCoords(TypedDict):
    str: Tuple[float, float, float]

class AtomicOrbital(TypedDict):
    atomic_orbital_number: int
    orbital_symbol: str
    energy: float


class AtomicContributions(TypedDict):
    atom_symbol: str
    atomic_orbitals: List[AtomicOrbital]


class MolecularOrbital(TypedDict):
    occupied: bool
    eigenvalue: float
    ### atomic_contribution = {"{atom_number}" : AtomicContributions}
    atomic_contributions: Dict[str, AtomicContributions]


def calc_atomic_weight(atomic_contribution: AtomicContributions) -> float:
    """
    the weight on each atom is the sum of the square of the coefficient
    for all atomic orbital centred on that atom.
    """
    atomic_orbitals = atomic_contribution["atomic_orbitals"]
    return sum(
        [a_orbital["energy"]**2 for a_orbital in atomic_orbitals]
    )


def calc_IPR(mo: MolecularOrbital):
    """
    Inverse Participation ratio:
    Measures how many atoms share the orbital.

    IPR = [ SIGMA_i {(WH_i)^-2} ]^-1
        IE for every atom i, calculate the square of the weight on that atom,
         sum them all together and take the inverse.

        in python: sum([weight(i)**-2 for i in orbital.atoms])**-1
    """

    return sum(
        [calc_atomic_weight(
            i)**-2 for i in mo["atomic_contributions"].values()]
    )**-1


def calc_weight_on_heteroatoms(mo: MolecularOrbital, heteroatom_symbol: str):
    """
    Weight on heteroatoms:
    One can define the weight of the orbital on N as:
        W_N = SIGMA_{i if S_i=N} { WH_i }
            IE: sum the weight for each atom i, where atomic symbol is N

            in python: sum([weight(i) for i in orbital.atoms if i.atomic_symbol == "N"]) 
    """
    return sum(
        [calc_atomic_weight(i) for i in mo["atomic_contributions"].values(
        ) if i["atom_symbol"].strip().lower() == heteroatom_symbol.strip().lower()]
    )

def calc_inertia_tensor(mo: MolecularOrbital, atom_coords: AtomicCoords):
    r"""
    See https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

    For rigid object of N point masses m_k,
    Components of moment of inertia tensor I_, are defined as:
    I_ij =def \Sigma_k=1^N { m_k * (||r_k||^2 kroneker_ij - x_i^(k) * x_j^(k) ) }
        where: 
        {i,j} are 1,2,3 referring to x,y,z respectively.
        r_k = [x_1^(k), x_2^(k), x_3^(k)] the vector to the point mass m_k
        kroneker_ij is the Kronecker delta (IE 1 if i==j else 0)

    Translating for use in Mol. orbitals:
        mass m_k will refer to the orbital weight on atom k.


    physical interpretation?
    I_xx is the moment of inertia around the x axis for things that are being rotated around the x axis
    and I_yx is the moment of inertia around the x axis for an object being rotated around the y axis.


    """

    def tensor_element(i, j):
        total = 0
        for atom_number, atomic_contribution in mo["atomic_contributions"].items():
            m_k = calc_atomic_weight(atomic_contribution)
            x, y, z = atom_coords[atom_number]
            ijmap = {
                1: x,
                2: y,
                3: z
            }
            if i==j:
                rhs = x**2 + y**2 + z**2 - ijmap[i]**2
            else:
                rhs = - (ijmap[i] * ijmap[j])
            result = m_k * rhs
            total += result
        return total

    return np.asarray([
        [tensor_element(1,1), tensor_element(1,2), tensor_element(1,3),]
        , [tensor_element(2,1), tensor_element(2,2), tensor_element(2,3),]
        , [tensor_element(3,1), tensor_element(3,2), tensor_element(3,3)]
    ])


def runtests():
    atomic_orbitals: List[AtomicOrbital] = [
        {
            "atomic_orbital_number": 1,
            "orbital_symbol": "1s",
            "energy": 1.0
        }
    ]
    atomic_contributions: Dict[str,AtomicContributions] = {
        "1": {
            "atom_symbol":"O",
            "atomic_orbitals": atomic_orbitals
            },
    }
    mo_sample: MolecularOrbital = {
        "occupied": True,
        "eigenvalue": 1,
        "atomic_contributions": atomic_contributions
    }
    expected_weight = 1
    calculated_weight = calc_atomic_weight(atomic_contributions["1"])
    assert expected_weight == calculated_weight
    exp_weight_on_O = 1
    calc = calc_weight_on_heteroatoms(mo_sample, "O")
    assert exp_weight_on_O == calc



if __name__ == "__main__":
    # lets run a little test here
    runtests()
    import sys
    import json
    # Pass the orbitals file as first argument
    orbital_file = sys.argv[1]
    with open(orbital_file, 'r') as JsonFile:
        content = json.load(JsonFile)
    homo: MolecularOrbital = content["54"]
    atom_coords: AtomicCoords = content["atomic_coords"]

    print(f"""

    input file {sys.argv[1]}

    inverse participation ratio = {calc_IPR(homo)}
    (minimum is 1, maximum is number of atoms)
    ----------------------------------------------
    weight on Nitrogen (N) = {calc_weight_on_heteroatoms(homo, "N")}
    weight on Oxygen (O) = {calc_weight_on_heteroatoms(homo, "O")}
    weight on Oxygen (C) = {calc_weight_on_heteroatoms(homo, "C")}
    weight on Oxygen (H) = {calc_weight_on_heteroatoms(homo, "H")}

    ----------------------------------------------

    Inertia tensor is:
    {calc_inertia_tensor(homo, atom_coords)}

    """)
