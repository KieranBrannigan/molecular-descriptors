# ref: https://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
from typing import List
from enum import Enum

from typing_extensions import Literal
from y4_python.python_modules.util import mol_from_smiles
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAliphaticCarbocycles
    , CalcNumAliphaticHeterocycles
    , CalcNumAliphaticRings
    , CalcNumAmideBonds
    , CalcNumAromaticHeterocycles
    , CalcNumAromaticRings
    , CalcNumAtomStereoCenters
    , CalcNumLipinskiHBA
    , CalcNumLipinskiHBD
    , CalcNumRotatableBonds
    , CalcTPSA
)
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

def clean_smiles(smiles: str):
    return "".join(x for x in smiles if x not in "(){}")

def num_bonds_to_atom(smiles: str, atom_symbol: str, bonded_to_symbol: str, bond_type):
    ""
    count: int = 0

    mol = mol_from_smiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == atom_symbol:
            for neigh in atom.GetNeighbors():
                count += neigh.GetSymbol() == bonded_to_symbol and mol.GetBondBetweenAtoms(atom.GetIdx(),neigh.GetIdx()).GetBondType() == bond_type
    return count

def num_of_phosphate_bonds(smiles: str):
    "Return the number of phosphate bonds (P=O) in the given smiles str."
    return num_bonds_to_atom(smiles, "P", "O", BondType.DOUBLE)

def num_of_sulfate_bonds(smiles: str):
    "Return the number of sulfate bonds (S=O) in the given smiles str."

    return num_bonds_to_atom(smiles, "S", "O", BondType.DOUBLE)

def num_of_atoms(smiles: str, atoms: List[str]) -> int:
    """
    Given a list of atoms return the number of occurances of those atoms.

    e.g. if atoms == ["F", "Cl", "Br", "I"] this will return the number of
        halides in the smiles.
    """
    return sum((smiles.lower().count(x.lower()) for x in atoms))