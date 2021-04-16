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

from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union
import json
import math
from itertools import combinations

from matplotlib.figure import Figure

from typing_extensions import TypedDict

import numpy as np
from numpy import linalg
from numpy.linalg import eig, norm

import matplotlib.colors as mcolors


from .util import scale_array

class PointMass(NamedTuple):
    mass: float
    coords: np.ndarray

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


class SerializedMolecularOrbital(TypedDict):
    """
    Holds some key information for molecular orbitals.

    Created using MolecularOrbital.toDict()
    """
    molecule_name: str
    centre_of_mass: Sequence[float]
    principal_moments: Sequence[float]
    principal_axes: Sequence[Sequence[float]]
    percent_on_N: float
    percent_on_O: float
    percent_on_S: float
    percent_on_P: float
    radial_distribution: List[float]

class MolecularOrbital:
    HOMO: int = -1
    LUMO: int = -2

    def __init__(self, mo: MolecularOrbitalDict, atomic_coords: AtomicCoords, molecule_name:str="N/A", mo_number:int=0, weight_scaling_factor=1, radial_distribution_kwargs: dict = {}):
        self.mo = mo
        self.atomic_coords = atomic_coords
        self._masses: Optional[List[PointMass]] = None
        self._center_of_mass: Optional[np.ndarray] = None
        self._inertia_tensor: Optional[np.ndarray] = None
        self._principal_axes: Optional[np.ndarray] = None
        self._principal_moments: Optional[np.ndarray] = None
        self._percent_on_N: Optional[float] = None
        self._percent_on_O: Optional[float] = None
        self._percent_on_S: Optional[float] = None
        self._percent_on_P: Optional[float] = None
        self.molecule_name = molecule_name
        self.mo_number = mo_number

        ### see self.calc_masses
        self.weight_scaling_factor = weight_scaling_factor

        self.radial_distribution_kwargs = radial_distribution_kwargs
        self._radial_distribution: Optional[List[float]] = None

    @property
    def radial_distribution(self) -> List[float]:
        ""
        if self._radial_distribution is None:
            X, self._radial_distribution = self.radial_dist_func(**self.radial_distribution_kwargs)
        return self._radial_distribution

    @property
    def masses(self) -> List[PointMass]:
        if self._masses is None:
            self._masses = self.calc_masses()
        return self._masses

    @property
    def center_of_mass(self) -> np.ndarray:
        if self._center_of_mass is None:
            self._center_of_mass = self.calc_center_of_mass()
        return self._center_of_mass

    @property
    def inertia_tensor(self) -> np.ndarray:
        if self._inertia_tensor is None:
            self._inertia_tensor = self.calc_inertia_tensor()
        return self._inertia_tensor

    @property
    def principal_axes(self) -> np.ndarray:
        if self._principal_axes is None:
            self.calc_principal_moments()
        return self._principal_axes # type: ignore  - self._principal_axes is set to ndarray in calc_principal_moments()

    @property
    def principal_moments(self) -> np.ndarray:
        if self._principal_moments is None:
            self.calc_principal_moments()
        return self._principal_moments # type: ignore  - self._principal_moments is set to ndarray in calc_principal_moments()

    @property
    def percent_on_N(self) -> float:
        if self._percent_on_N is None:
            self._percent_on_N = self.calc_weight_on_heteroatom("N")
        return self._percent_on_N

    @property
    def percent_on_O(self) -> float:
        if self._percent_on_O is None:
            self._percent_on_O = self.calc_weight_on_heteroatom("O")
        return self._percent_on_O

    @property
    def percent_on_S(self) -> float:
        if self._percent_on_S is None:
            self._percent_on_S = self.calc_weight_on_heteroatom("S")
        return self._percent_on_S

    @property
    def percent_on_P(self) -> float:
        if self._percent_on_P is None:
            self._percent_on_P = self.calc_weight_on_heteroatom("P")
        return self._percent_on_P

    @classmethod
    def fromJsonFile(cls, orbital_file: str, mo_number: int, **kwargs) -> 'MolecularOrbital':
        with open(orbital_file, 'r') as JsonFile:
            content = json.load(JsonFile)

        if mo_number in [cls.HOMO, cls.LUMO]:
            homo_num, lumo_num = cls.homoLumoNumbersFromJson(content)
            if mo_number == cls.HOMO:
                mo_number = homo_num
            elif mo_number == cls.LUMO:
                mo_number = lumo_num


        homo_dict: MolecularOrbitalDict = content[str(mo_number)]
        atom_coords: AtomicCoords = content["atomic_coords"]

        return cls(homo_dict, atom_coords, mo_number=mo_number, **kwargs)

    @staticmethod
    def homoLumoNumbersFromJson(orbital_file_content: dict) -> Tuple[int, int]:

        keys = list(orbital_file_content.keys())
        try: keys.remove("atomic_coords") # ValueError if doesn't exist, which is fine.
        except: pass
        for idx, mo_number in enumerate(keys):
            if not orbital_file_content[mo_number]["occupied"]:
                HOMO_num = int(keys[idx-1])
                LUMO_num = int(keys[idx])
                break
        else: # didn't break
            HOMO_num = int(keys[idx]) # type:ignore - I want it to throw if we don't get numbers
            LUMO_num = False

        return (HOMO_num, LUMO_num) # type:ignore - I want it to throw if we don't get numbers

    def serialize(self) -> str:
        """
        Create a dict with important calculated properties and then convert to json str with json module.
        """
        
        return json.dumps(self.toDict())

    def toDict(self) -> SerializedMolecularOrbital:
        return {
            "molecule_name": self.molecule_name
            , "centre_of_mass": self.center_of_mass.tolist()
            , "principal_moments": self.principal_moments.tolist()
            , "principal_axes": self.principal_axes.tolist()
            , "percent_on_N": self.percent_on_N
            , "percent_on_O": self.percent_on_O
            , "percent_on_S": self.percent_on_S
            , "percent_on_P": self.percent_on_P
            , "radial_distribution": self.radial_distribution
        } # type: ignore

    def radial_dist_func(self, r_min=0.8, r_max=10, r_step=0.1, sigma=0.1) -> Tuple[List[float], List[float]]:
        """
        f(r) = SUM_{i<j} { w_i * w_j * g( r-r_{ij} ) }
        where g(r) is a Gaussian function with arbitrary standard deviation `sigma`, i.e.
        
                       1                    -r^2
        g(r) =  ---------------- * exp(  -------------   ) 
                sigma*sqrt(2*pi)          2 * sigma^2
        
        If Two orbitals have similar shape then corresponding functions f_1(r) and f_2(r) are similar.
        Does not depend on orientation.

        In practice one can use a sample of the function for r between ~0.8 and ~10 angstrom 
        in steps of ~0.1 angstroms and sigma set to ~0.1. 
        This will associate to any orbital a vector of ~90 elements. 
        The distance between two orbitals can be computed as the norm of the 
        difference between the two vectors.
        
        This function will call f(r) for all values between r_min and r_max in steps of r_step.
        sigma is also given as an argument.
        The default arg values will give a vector with 92 elements ( r_max-r_min / r_step )

        """


        # pre-calculating the w_i * w_j and r_{ij} terms
        weight_products = []
        mass_distances = []
        for i_mass, j_mass in combinations(self.masses, 2):
            # iterate i<j
            "do something with i_mass and j_mass"
            weight_products.append(i_mass.mass * j_mass.mass)
            mass_distances.append( linalg.norm(i_mass.coords - j_mass.coords) )

        g_norm = 1 / ( sigma*math.sqrt(2*math.pi) )
        g_const = 1 / ( 2* (sigma**2) )
        def g(r) -> float:
            "Gaussian function with arbitrary st dev, sigma"
            return g_norm * math.exp( - (r**2) * g_const )

        def f(r) -> float:
            "sum for i<j, w_i * w_j * g( r-r_{ij} )"
            return sum(
                weight_products[idx] * g(r - mass_distances[idx]) 
                    for idx in range(len(weight_products))
            )

        X: List[float] = []
        F: List[float] = []
        for r in np.arange(r_min, r_max, r_step):
            X.append(r)
            F.append(f(r))

        return X, F

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
            [
                self.calc_atomic_weight(i)**-2 
                    for i in self.mo["atomic_contributions"].values()
            ]
        )**-1


    def calc_weight_on_heteroatom(self, heteroatom_symbol: str) -> float:
        """
        Weight on heteroatoms:
        One can define the weight of the orbital on N as:
            W_N = SIGMA_{i if S_i=N} { WH_i }
                IE: sum the weight for each atom i, where atomic symbol is N

                in python: sum([weight(i) for i in orbital.atoms if i.atomic_symbol == "N"]) 
        """
        return float(sum(
            [
                self.calc_atomic_weight(i) for i in self.mo["atomic_contributions"].values() 
                    if i["atom_symbol"].strip() == heteroatom_symbol.strip()
            ]
        ))

    def calc_inertia_tensor(self) -> np.ndarray:
        r"""
        See https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor

        For rigid object of N point masses m_k

        Translating for use in Mol. orbitals:
            mass m_k will refer to the orbital weight on atom k.

        When calculated, we should calculate relative to the center of mass, IE
        we should translate the positions of all masses by -[center_of_mass]
        
        or: position relative to centre of mass, r_cm = r - cm  # vectors
        where: r = position relative to origin; cm = position of center of mass
        """
        def fun(pm: PointMass) -> PointMass:
            return PointMass(mass=pm.mass, coords=(pm.coords-self.center_of_mass))
        
        masses_relative_to_CM = list(map(fun, self.masses))

        ### pass to calc_inertia_tensor
        return calc_inertia_tensor(masses_relative_to_CM)


    def calc_masses(self) -> List[PointMass]:
        def mapfun(atom_num: str):
            atomic_contribution: AtomicContribution = self.mo["atomic_contributions"][atom_num]
            return PointMass(
                mass=self.calc_atomic_weight(atomic_contribution) ** self.weight_scaling_factor,
                coords=np.array(self.atomic_coords[atom_num])
                )

        return list(map(mapfun, self.mo["atomic_contributions"].keys()))

    def calc_center_of_mass(self) -> np.ndarray:

        return calc_center_of_mass(self.masses)
    
    def calc_principal_moments(self):
        """
        Check if self.inertiaTensor is calculated. If not, calculate it (self.calc_inertia_tensor).
        Then -> self.principalMoments, self.principalAxes = eig(self.inertiaTensor)
        """

        self._principal_moments, self._principal_axes = eig(self.inertia_tensor)

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

        vectors = self.principal_axes
        moments = self.principal_moments
        cx,cy,cz = self.center_of_mass # type:ignore
        
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

    def get_atom_plot_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:

        xs = []
        ys = []
        zs = []
        weights = []
        colours = []
        atom_numbers = []
        atomic_contribution: AtomicContribution
        for atom_number, atomic_contribution in self.mo["atomic_contributions"].items():
            x,y,z = self.atomic_coords[atom_number]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            weight = self.calc_atomic_weight(atomic_contribution) ** self.weight_scaling_factor
            weights.append(
                weight
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

def calc_principal_axes(inertiaTensor):
    return eig(inertiaTensor)


def calc_center_of_mass(masses: List[PointMass]) -> np.ndarray:
    """
    X_cm = sum( mass*position for point in masses ) / total mass

    easier as: sum( (mass/totalmass) * position for point in masses)

    TODO: optimise by using numpy arrays.

    """
    from functools import reduce

    ### TODO: The total mass is almost 1 (0.9999989...) Should I just set it to one, since this is probably a rounding/floating point error?
    total_mass = sum((x.mass for x in masses))

    map_ = map(lambda x: x.mass * x.coords / total_mass, masses)
    sum_ = sum(np.array(list(map_)))
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
            x, y, z = coords # type: ignore
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
