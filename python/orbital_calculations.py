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

from typing import Dict, List
import json

from typing_extensions import TypedDict


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
    atomic_contributions: Dict[str, AtomicContributions]


def calc_atomic_weight(atomic_contribution) -> float:
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

    print(f"""
    inverse participation ratio = {calc_IPR(homo)}
    (minimum is 1, maximum is number of atoms)
    ----------------------------------------------
    weight on Nitrogen (N) = {calc_weight_on_heteroatoms(homo, "N")}
    weight on Oxygen (O) = {calc_weight_on_heteroatoms(homo, "O")}
    weight on Oxygen (C) = {calc_weight_on_heteroatoms(homo, "C")}
    weight on Oxygen (H) = {calc_weight_on_heteroatoms(homo, "H")}
    """)
