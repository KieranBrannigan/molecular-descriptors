# ref: https://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
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

def clean_smiles(smiles: str):
    return "".join(x for x in smiles if x not in "(){}")

def get_end_of_branch_idx(smiles: str, start_of_branch_idx: int):
    """
    Given index where a branch starts in a smiles string
    , return the idx where the branch ends.
    """
    # keep track of open brackets.
    # ( = +1
    # ) = -1
    # when this becomes 0, branch is closed, return idx
    bracket_count = 1
    for idx in range(start_of_branch_idx+1, len(smiles)):
        char = smiles[idx]
        if char == "(":
            bracket_count += 1
        elif char == ")":
            bracket_count -= 1

        if bracket_count < 1:
            return idx

    raise ValueError(f"End of branch not found.\nSMILES={smiles}\nstart_of_branch_idx={start_of_branch_idx}")

def check_bonded_to_group(smiles: str, idx_of_atom_to_check: int, group: str) -> int:
    """
    Given smiles representation of molecule, and the idx of an atom in that smiles, 
    and a given group (substring of a smiles molecule) to check:
    return True if that atom is bonded to the specified group,
    else return False.

    e.g. if group is `=O` then we are checking if atom is doublebonded to oxygen

    this function checks for branching, so don't include any branches in group
    """
    def inner(smiles: str, idx_of_atom_to_check: int, group: str) -> int:
        idx = idx_of_atom_to_check
        count = 0
        # now we're cooking with gas
        if smiles[idx+1:idx+len(group)+1] == group: # straight up bonded to group
            return True
        elif smiles[idx+1] == "(": # start of branching 
            if smiles[idx+2:idx+2+len(group)] == group: # straight up bonded to group
                #return True
                count+=1
            # else: # go to end of branch and recurse
            #     end_of_branch_idx = get_end_of_branch_idx(smiles, idx+1)
            #     return inner(smiles, end_of_branch_idx, group)
            end_of_branch_idx = get_end_of_branch_idx(smiles, idx+1)
            count+=inner(smiles,end_of_branch_idx, group)
        # else: # not bonded to group or a branch, therefore return False.
        #     return False
        return count

    ### Checks forwards and backwards

    return inner(smiles, idx_of_atom_to_check, group) + inner(smiles[::-1], len(smiles)-1-idx_of_atom_to_check, group)


def num_of_bonds_to_group(smiles: str, atom_symbol: str, group: str):
    """
    Return the number of -ate bonds (e.g. P=O if atom_symbol == "P") in the given SMILES string.

    Atom symbol is the atom to check.
    """

    count = 0

    for idx in range(len(smiles)):
        char = smiles[idx]
        if char == atom_symbol:
            count += check_bonded_to_group(smiles, idx, group)

    return count

def num_of_phosphate_bonds(smiles: str):
    "Return the number of phosphate bonds (P=O) in the given smiles str."

    return num_of_bonds_to_group(smiles, "P", "=O")

def num_of_sulfate_bonds(smiles: str):
    "Return the number of sulfate bonds (S=O) in the given smiles str."

    return num_of_bonds_to_group(smiles, "S", "=O")