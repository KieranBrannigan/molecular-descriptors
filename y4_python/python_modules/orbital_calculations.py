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

from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union
import json
from matplotlib.figure import Figure

from typing_extensions import TypedDict

import numpy as np
from numpy.linalg import eig

import matplotlib.colors as mcolors


from .util import scale_array

class PointMass(NamedTuple):
    mass: float
    coords: Tuple[float, float, float]

class AtomicCoords(TypedDict):
    ### TODO check: is this { "atom_num": (x,y,z) } ??
    """
    This isn't quite right, this is suggesting atomcoord: AtomicCoords = {"str": (0.1,0.2,0.1)}
    """
    str: Tuple[float, float, float]

class AtomicOrbital(TypedDict):
    atomic_orbital_number: int
    orbital_symbol: str
    energy: float


class AtomicContribution(TypedDict):
    atom_symbol: str
    atomic_orbitals: List[AtomicOrbital]


class MolecularOrbitalDict(TypedDict):
    occupied: bool
    eigenvalue: float
    ### atomic_contribution = {"{atom_number}" : AtomicContribution}
    atomic_contributions: Dict[str, AtomicContribution]

class MolecularOrbital:

    def __init__(self, mo: MolecularOrbitalDict, atomic_coords: AtomicCoords):
        self.mo = mo
        self.atomic_coords = atomic_coords
        self._masses: Optional[List[PointMass]] = None
        self._center_of_mass: Optional[Tuple[float, float, float]] = None
        self._inertia_tensor: Optional[np.ndarray] = None
        self._principle_axes: Optional[np.ndarray] = None
        self._principle_moments: Optional[np.ndarray] = None

    @property
    def masses(self) -> List[PointMass]:
        if not self._masses:
            self._masses = self.calc_masses()
        return self._masses

    @property
    def center_of_mass(self) -> Tuple[float, float, float]:
        if not self._center_of_mass:
            self._center_of_mass = self.calc_center_of_mass()
        return self._center_of_mass

    @property
    def inertia_tensor(self) -> np.ndarray:
        if not self._inertia_tensor:
            self._inertia_tensor = self.calc_inertia_tensor()
        return self._inertia_tensor

    @property
    def principle_axes(self) -> np.ndarray:
        if not self._principle_axes:
            self.calc_principle_moments()
        return self._principle_axes # type: ignore  - self._principle_axes is set to ndarray in calc_principle_moments()

    @property
    def principle_moments(self) -> np.ndarray:
        if not self._principle_moments:
            self.calc_principle_moments()
        return self._principle_moments # type: ignore  - self._principle_moments is set to ndarray in calc_principle_moments()

    @classmethod
    def fromJsonFile(cls, orbital_file: str, mo_number: int) -> 'MolecularOrbital':
        with open(orbital_file, 'r') as JsonFile:
            content = json.load(JsonFile)
        homo_dict: MolecularOrbitalDict = content[str(mo_number)]
        atom_coords: AtomicCoords = content["atomic_coords"]

        return cls(homo_dict, atom_coords)

    @staticmethod
    def homoLumoNumbersFromJson(orbital_file: str) -> Tuple[int, int]:
        with open(orbital_file, 'r') as JsonFile:
            content: dict = json.load(JsonFile)
        keys = list(content.keys())

        for idx, mo_number in enumerate(keys):
            if mo_number == "atomic_coords":
                continue
            if not content[mo_number]["occupied"]:
                HOMO_num = int(keys[idx-1])
                LUMO_num = int(keys[idx])

        return (HOMO_num, LUMO_num) # type:ignore

    
    def calc_atomic_weight(self, atomic_contribution: AtomicContribution) -> float:
        """
        the weight on each atom is the sum of the square of the coefficient
        for all atomic orbital centred on that atom.
        """
        atomic_orbitals = atomic_contribution["atomic_orbitals"]
        return sum(
            [a_orbital["energy"]**2 for a_orbital in atomic_orbitals]
        )


    def calc_IPR(self):
        """
        Inverse Participation ratio:
        Measures how many atoms share the orbital.

        IPR = [ SIGMA_i {(WH_i)^-2} ]^-1
            IE for every atom i, calculate the square of the weight on that atom,
            sum them all together and take the inverse.

            in python: sum([weight(i)**-2 for i in orbital.atoms])**-1
        """

        return sum(
            [self.calc_atomic_weight(
                i)**-2 for i in self.mo["atomic_contributions"].values()]
        )**-1


    def calc_weight_on_heteroatoms(self, heteroatom_symbol: str):
        """
        Weight on heteroatoms:
        One can define the weight of the orbital on N as:
            W_N = SIGMA_{i if S_i=N} { WH_i }
                IE: sum the weight for each atom i, where atomic symbol is N

                in python: sum([weight(i) for i in orbital.atoms if i.atomic_symbol == "N"]) 
        """
        return sum(
            [self.calc_atomic_weight(i) for i in self.mo["atomic_contributions"].values(
            ) if i["atom_symbol"].strip().lower() == heteroatom_symbol.strip().lower()]
        )

    def calc_inertia_tensor(self) -> np.ndarray:
        r"""
        See https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

        For rigid object of N point masses m_k

        Translating for use in Mol. orbitals:
            mass m_k will refer to the orbital weight on atom k.
        """

        ### pass to calc_inertia_tensor
        return calc_inertia_tensor(self.masses)


    def calc_masses(self) -> List[PointMass]:

        def mapfun(atom_num: str):
            atomic_contribution: AtomicContribution = self.mo["atomic_contributions"][atom_num]
            return PointMass(
                mass=self.calc_atomic_weight(atomic_contribution),
                coords=self.atomic_coords[atom_num]
                )

        return list(map(mapfun, self.mo["atomic_contributions"].keys()))

    def calc_center_of_mass(self) -> Tuple[float, float, float]:

        return calc_center_of_mass(self.masses)
    
    def calc_principle_moments(self):
        """
        Check if self.inertiaTensor is calculated. If not, calculate it (self.calc_inertia_tensor).
        Then -> self.principleMoments, self.principleAxes = eig(self.inertiaTensor)
        """

        self.principleMoments, self.principleAxes = eig(self.inertia_tensor)

    def plot(self, mol_name, axis_number, fig:Figure):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(axis_number, projection="3d") # type:ignore
        #ax.set_axis_off()
        ax.set_facecolor("slategrey")
        ax.set_title(mol_name)
        xs, ys, zs, weights, scaledWeights, colours, atom_numbers = self.get_atom_plot_values()
        scatter = ax.scatter(xs, ys, zs=zs, s=scaledWeights, c=colours, depthshade=True)

        # for idx, num in enumerate(atom_numbers):
        #     ax.text(
        #         xs[idx], ys[idx], zs[idx]
        #         , f"n={num}"#\nw={np.round(weights[idx], decimals=4)}"
        #         , color='green'
        #     )

        vectors = self.principle_axes
        moments = self.principle_moments
        cx,cy,cz = self.center_of_mass
        
        for idx, vector in enumerate(vectors): # type:ignore - I'm pretty sure np.ndarray is an iterable
            #moment = moments[idx]
            X, Y, Z = vector
            ax.quiver(cx, cy, cz, X, Y, Z)
            ax.quiver(cx, cy, cz, -X, -Y, -Z)
            ax.text(
                cx+X, cy+Y, cz+Z
                , idx+1
                , color="purple"
            )

        return ax

    def get_atom_plot_values(self):

        xs = []
        ys = []
        zs = []
        ### TODO: scale the weights between a min and max value
        weights = []
        colours = []
        atom_numbers = []
        atomic_contribution: AtomicContribution
        for atom_number, atomic_contribution in self.mo["atomic_contributions"].items():
            x,y,z = self.atomic_coords[atom_number]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            self.calc_atomic_weight(atomic_contribution)
            weights.append(
                self.calc_atomic_weight(atomic_contribution)
            )
            colours.append(
                self.get_colour_from_atomsymbol(atomic_contribution["atom_symbol"])
            )
            atom_numbers.append(atom_number)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        colours = np.array(colours)
        weights = np.array(weights)
        scaledWeights = scale_array(weights, min_=0, max_=1000)
        return xs, ys, zs, weights, scaledWeights, colours, atom_numbers

    def get_colour_from_atomsymbol(self, atom_symbol):
        "Convert the atom_symbol to its colour (TODO: in hex?)"
        mapping = {
            "C": "black"
            , "H": "white"
            , "O": "firebrick"
            , "N": "royalblue"
        }
        color = mapping.get(atom_symbol, "yellow")
        return mcolors.to_rgb(color)

def calc_principle_axes(inertiaTensor):
    return eig(inertiaTensor)

def vecAdd(v1, v2):
    return tuple((v1[idx] + v2[idx] for idx in range(len(v1))))

def calc_center_of_mass(masses: List[PointMass]) -> Tuple[float, float, float]:
    """
    X_cm = sum( mass*position for point in masses ) / total mass

    easier as: sum( (mass/totalmass) * position for point in masses)

    TODO: optimise by using numpy arrays.

    """
    from functools import reduce

    total_mass = sum((x.mass for x in masses))

    X_cm = [tuple(map(lambda coord: x.mass * coord / total_mass, x.coords)) for x in masses]
    sum_ = reduce(vecAdd, X_cm)
    return sum_
    

def calc_inertia_tensor(masses: List[PointMass]) -> np.ndarray:
    """
    given a list of point masses, with mass m and coords (x,y,z), calculate and return
    the inertia tensor.

    See https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

    For rigid object of N point masses m_k,
    Components of moment of inertia tensor I_, are defined as:
    I_ij =def \Sigma_k=1^N { m_k * (||r_k||^2 kroneker_ij - x_i^(k) * x_j^(k) ) }
        where: 
        {i,j} are 1,2,3 referring to x,y,z respectively.
        r_k = [x_1^(k), x_2^(k), x_3^(k)] the vector to the point mass m_k
        kroneker_ij is the Kronecker delta (IE 1 if i==j else 0)

    physical interpretation?
    I_xx is the moment of inertia around the x axis for things that are being rotated around the x axis
    and I_yx is the moment of inertia around the x axis for an object being rotated around the y axis.
    """
    ijmap = {
        1: "x",
        2: "y",
        3: "z"
    }
    def tensor_element(i, j):
        total = 0
        for mass, coords in masses:
            x, y, z = coords
            if i==j:
                i_xyz = locals()[ijmap[i]]
                rhs = x**2 + y**2 + z**2 - i_xyz**2
            else:
                i_xyz = locals()[ijmap[i]]
                j_xyz = locals()[ijmap[j]]
                rhs = - (i_xyz * j_xyz)
            result = mass * rhs
            total += result
        return total

    return np.array([
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
    atomic_contributions: Dict[str,AtomicContribution] = {
        "1": {
            "atom_symbol":"O",
            "atomic_orbitals": atomic_orbitals
            },
    }
    mo_sample = MolecularOrbital({
        "occupied": True,
        "eigenvalue": 1,
        "atomic_contributions": atomic_contributions
    }, atomic_coords={"str": (1,2,3)})
    expected_weight = 1
    calculated_weight = mo_sample.calc_atomic_weight(atomic_contributions["1"])
    assert expected_weight == calculated_weight
    exp_weight_on_O = 1
    calc = mo_sample.calc_weight_on_heteroatoms("O")
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
    homo_dict: MolecularOrbitalDict = content["54"]
    atom_coords: AtomicCoords = content["atomic_coords"]

    homo = MolecularOrbital(homo_dict, atom_coords)

    print(f"""

    input file {sys.argv[1]}

    inverse participation ratio = {homo.calc_IPR()}
    (minimum is 1, maximum is number of atoms)
    ----------------------------------------------
    weight on Nitrogen (N) = {homo.calc_weight_on_heteroatoms("N")}
    weight on Oxygen (O) = {homo.calc_weight_on_heteroatoms("O")}
    weight on Oxygen (C) = {homo.calc_weight_on_heteroatoms("C")}
    weight on Oxygen (H) = {homo.calc_weight_on_heteroatoms("H")}

    ----------------------------------------------

    Inertia tensor is:
    {homo.inertia_tensor}

    """)
