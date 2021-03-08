import os
from itertools import combinations
from pprint import pprint

from .python_modules.orbital_calculations import MolecularOrbital

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

def reducefname(filename):
        molname: str = os.path.splitext(os.path.basename(filename))[0]
        return molname.replace("_output","").upper()

def logfun(homo: MolecularOrbital):
    print(homo.molecule_name)
    print(f"HOMO number = {homo.mo_number}")
    print(f"centre of mass = {homo.center_of_mass}")
    print(f"principle moments: ")
    print(homo.principle_moments)
    print(f"principle axes: ")
    print(homo.principle_axes)

def compare2files(file1: str, file2: str):
    """
    Given two molecular orbital json files, print out certain MO values
    for comparison.
    """
    dpi = 80
    fig = plt.figure(figsize=(1600/dpi, 1600/dpi), dpi=dpi)
    file1_name = reducefname(os.path.basename(file1))
    file2_name = reducefname(os.path.basename(file2))
    ### Just comparing HOMO for now
    homo1 = MolecularOrbital.fromJsonFile(file1, MolecularOrbital.HOMO, molecule_name=file1_name, weight_scaling_factor=10)
    homo1_principle_moments = homo1.principle_moments
    homo1_principle_axes = homo1.principle_axes
    ax1 = homo1.plot(file1_name, 121, fig)

    homo2 = MolecularOrbital.fromJsonFile(file2, MolecularOrbital.HOMO, molecule_name=file2_name, weight_scaling_factor=10)
    homo2_principle_moments = homo2.principle_moments
    homo2_principle_axes = homo2.principle_axes
    ax2 = homo2.plot(file2_name, 122, fig)


    logfun(homo1)
    logfun(homo2)

    def update(i):
        ax1.view_init(10, i*2)
        ax2.view_init(10, i*2)
        return fig
    
    ### save fig
    # fig.tight_layout()
    # anim = FuncAnimation(fig, update, frames=180, interval=1)
    # anim.save(f"{file1_name}-vs-{file2_name}.gif", dpi=80, writer="imagemagick")


def main(directory):
    """
    directory contains the molecular orbitals json files

    For each pair in filepairs,
        run compare2files on pair
        IE: print the molname, mo_num, princ. moments, princ. axes
    """

    # filepairs = combinations(
    #     [x for x in os.listdir(directory) if x.endswith(".json") and not "ZINC" in x]
    #     , 2)

    filepairs = [
            ("naphthalene_output.json", "butyl_naphthalene_output.json")
            , ("anthracene_output.json", "butyl_anthracene_output.json")
            , ("naphthalene_output.json", "naphthalene_butyl_anthracene_output.json")
            , ("anthracene_output.json", "naphthalene_butyl_anthracene_output.json")
            , ("diphenyl_butadiene_output.json", "diphenyl_hexatriene_output.json")
    ]

    for file1, file2 in filepairs:
        print(f"Compare {reducefname(file1)} and {reducefname(file2)}\n".upper())
        joined_file1 = os.path.join(directory, file1)
        joined_file2 = os.path.join(directory, file2)
        compare2files(joined_file1, joined_file2)

        print("\n----------------------------------------------\n")

    plt.show()

if __name__ == "__main__":
    main(
        os.path.join("..","sampleInputs","PM7_orbitals")
    )