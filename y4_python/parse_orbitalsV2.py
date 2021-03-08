"""

Given an input gaussian log file, parse out the atomic coordinates and molecular orbitals.

Output:
    
    orbital table/matrix:
        Example:
                 Molecular Orbital Coefficients:
                                        1         2         3         4         5
                                        O         O         O         O         O
                  Eigenvalues --    -1.36363  -1.17864  -1.15298  -1.11813  -1.11310
                1 1   O  1S          0.25305  -0.05726   0.00090   0.56130   0.01346
                2        1PX         0.18053  -0.02074   0.00605   0.11692   0.01466
                3        1PY         0.14800  -0.00818  -0.00997   0.14767  -0.02255
                4        1PZ         0.14746  -0.02153   0.00477   0.08341   0.02417
                5 2   N  1S          0.78611  -0.03071   0.00727   0.01200   0.03001     

    
    molname_Molecular_orbitals.json: [
        {
            MO_number : int
            , occupied : bool
            , eigenvalue : float
            , atomic_contributions: [
                {
                    atom_number : int
                    , atom_symbol : str
                    , atomic_orbitals : [
                        orbital_symbol : str
                        , energy : float
                    ]
                },
                ...
            ]
        }
        , ...
    ]


    HOMO MO_number given by MO_number of last occupied MO
    LUMO MO_number given by HOMO MO_number + 1

"""

import sys
import os
from typing import Iterable, List
import re
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input-file", dest="inputFile", help="input gaussian log file to parse orbitals from", type=str
)
parser.add_argument(
    "--orbitals", type=str, dest="orbitals", help='string of comma separated values of the orbitals to extract. Values can be an integer or the string "homo" or "lumo". Leave blank to extract all orbitals or don\'t pass this parameter.'
)

args = parser.parse_args()

inFile: str = args.inputFile

orbitals = args.orbitals
if orbitals:
    orbitals = orbitals.split(",")
else:
    orbitals = [""]


molName, _ = os.path.splitext(inFile)

###########################
### Read the input log file
###########################

with open(inFile, "r") as ReadFile:
    fileContent: List[str] = list(ReadFile.readlines())

##################################
### Get list of orbital symmetries 
##################################

### Natoms line regex
start_coords_pattern = re.compile(r"Standard orientation:")

occupied_symm_lines: List[str] = []
virtual_symm_lines: List[str] = []

start_idx = None
occ = None
virt = None
MO_table_start_idx = None
MO_table_end_idx = None
start_coords_idx = False
atomic_coords = {}
for idx, line in enumerate(fileContent):
    if "Orbital symmetries:" in line:
        start_idx = idx
    if start_idx:
        if "Occupied" in line:
            occ = True
        elif "Virtual" in line:
            occ = None
            virt = True
        if "electronic state" in line:
            end_idx = idx
            start_idx = None
            occ = None
            virt = None
        if occ:
            occupied_symm_lines.append(line.strip())
        elif virt:
            virtual_symm_lines.append(line.strip())

    elif start_coords_idx and idx >= start_coords_idx:
        if line.strip().replace("-","") == "":
            start_coords_idx = False
        else:
            atom_num, _, _, x, y, z = line.strip().split()
            ### {"{atom_num}" : [x,y,z]}
            atomic_coords[atom_num] = [float(i) for i in [x,y,z]]

    elif re.search(start_coords_pattern, line):
        start_coords_idx = idx+5

    elif "molecular orbital coefficients:" in line.lower():
        MO_table_start_idx = idx+1
        
    elif "Density Matrix:" in line:
        MO_table_end_idx = idx
        break


### Fix orbital symmetries
occupied_symm_lines[0] = occupied_symm_lines[0].split("Occupied  ")[1]
virtual_symm_lines[0] = virtual_symm_lines[0].split("Virtual   ")[1]

### flatten into lists of symmetries
occupied_symm = [item for sublist in map(lambda x: x.strip().split(" "), occupied_symm_lines) for item in sublist]
virtual_symm = [item for sublist in map(lambda x: x.strip().split(" "), virtual_symm_lines) for item in sublist]

number_occupied_orbitals: int = len(occupied_symm)
number_virtual_orbitals: int = len(virtual_symm)
#print(number_occupied_orbitals + number_virtual_orbitals)


### What orbitals are wanted?
if orbitals == ['']:
    wanted_orbitals = [x for x in range(number_occupied_orbitals + number_virtual_orbitals)]
elif len(orbitals) > 0:
    wanted_orbitals: List[int] = []
    for x in orbitals:
        if x.lower() == "homo":
            wanted_orbitals.append(number_occupied_orbitals)
        elif x.lower() == "lumo":
            wanted_orbitals.append(number_occupied_orbitals+1)
        else:
            wanted_orbitals.append(int(x))
else:
    raise Exception("something went wrong with wanted_orbitals")

####################################
### Grab the molecular orbital table
####################################

data = {}

def saveData(MO_numbers
            , isOccupied
            , eigenvalues
            , atomic_data):
    """
    molname_Molecular_orbitals.json --> {
        MO_number : {            
            , occupied : bool
            , eigenvalue : float
            , atomic_contributions: [
                atom_number : {
                    , atom_symbol : str
                    , atomic_orbitals : [
                        {
                            atomic_orbital_number : int
                            , orbital_symbol : str
                            , energy : float
                        }
                        , ...
                    ]
                },
                ...
            ]
        }
        , ...
    }
    """
    MO_data = {

    }

    ### Process atomic data
    atomic_data_T = np.array(atomic_data).T
    for idx, MO_number in enumerate(MO_numbers):
        MO_data[MO_number] = {
            "occupied" : isOccupied[idx]
            , "eigenvalue": eigenvalues[idx]
            , "atomic_contributions" : {}
        }
    for row in atomic_data:
        (
            ao_number
            , atom_number
            , atom_symbol
            , ao_symbol
            , a_contributions
        ) = row
        for idx, energy in enumerate(a_contributions):
            mo_num = MO_numbers[idx]
            
            if atom_number not in MO_data[mo_num]["atomic_contributions"]:
                MO_data[mo_num]["atomic_contributions"][atom_number] = {}
            
            if "atom_symbol" not in MO_data[mo_num]["atomic_contributions"][atom_number]:
                MO_data[mo_num]["atomic_contributions"][atom_number]["atom_symbol"] = atom_symbol
            
            if "atomic_orbitals" not in MO_data[mo_num]["atomic_contributions"][atom_number]:
                MO_data[mo_num]["atomic_contributions"][atom_number]["atomic_orbitals"] = [{
                    "atomic_orbital_number" : ao_number
                    , "orbital_symbol" : ao_symbol
                    , "energy" : energy
                }]
            else:
                MO_data[mo_num]["atomic_contributions"][atom_number]["atomic_orbitals"].append({
                    "atomic_orbital_number" : ao_number
                    , "orbital_symbol" : ao_symbol
                    , "energy" : energy
                })



    data.update(MO_data)

def any_in(l1: List, l2:List):
    for el in l1:
        if el in l2:
            return True
    return False 

### WE ARE ASSUMING THAT the Number of MO orbitals will be the number of atomic orbitals
if not MO_table_start_idx:
    raise Exception("No start index for MO table")
if not MO_table_end_idx:
    raise Exception("No end index for MO table")

### Define regex patterns
m_orb_num_pattern = re.compile(r" {5,}([\d ][\d ]\d +){1,5}")
eigenvalues_pattern = re.compile(r"     Eigenvalues --    ")
isOccupied_pattern = re.compile(r" {5,}([^\d\W] {9})+")
a_orbital_pattern = re.compile(r"^[\d ]{1,3}\d [\d ]{1,3} [\w ]{2} \d\w[\w ] ")


MO_table_lines = fileContent[MO_table_start_idx:MO_table_end_idx]

MO_numbers = None
eigenvalues = None
isOccupied = None
atomic_data = []
for line in MO_table_lines:

    ### Match MO number line
    if m_orb_num_pattern.match(line):
        if MO_numbers:
            ### TODO: save data from this chunk before starting next chunk
            saveData(MO_numbers, isOccupied, eigenvalues, atomic_data)
        atomic_data = []
        
        MO_numbers = [int(x) for x in line.strip().split()]
        if not any_in(MO_numbers, wanted_orbitals):
            MO_numbers = None

    ### Match eigenvalues line
    elif eigenvalues_pattern.match(line) and MO_numbers:
        eigenvalues = [float(x) for x in line[23:].split()]

    ### Match isOccupied line
    elif isOccupied_pattern.match(line) and MO_numbers:
        isOccupied = [True if x.lower() == "o" else False for x in line.strip().split()]
        pass

    ### Match atomic orbital contribution line
    elif a_orbital_pattern.match(line) and MO_numbers:
        ### TODO: split into relevant columns of information
        ao_number = int(line[0:4])
        tmp_atom_num = line[5:8]
        if tmp_atom_num.strip() == '':
            tmp_atom_num = atom_num
        else:
            tmp_atom_num = int(tmp_atom_num)
            atom_num = int(tmp_atom_num)
        tmp_atom_symbol = line[9:11]
        if tmp_atom_symbol.strip() == "":
            tmp_atom_symbol = atom_symbol
        else:
            tmp_atom_symbol = tmp_atom_symbol.strip()
            atom_symbol = tmp_atom_symbol
        ao_symbol = line[12:15].strip()
        energies = [float(x) for x in line[16:].split()]
        row = [ao_number, tmp_atom_num, tmp_atom_symbol, ao_symbol, energies]
        atomic_data.append(row)

### TODO: save the last chunk
if MO_numbers:
    saveData(MO_numbers, isOccupied, eigenvalues, atomic_data)

#############################
# Save atomic coordinates
#############################
data["atomic_coords"] = atomic_coords

json.dump(data, sys.stdout)