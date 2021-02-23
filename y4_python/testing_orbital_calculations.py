import os
from itertools import combinations
from pprint import pprint

from .python_modules.orbital_calculations import MolecularOrbital

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

def reducefname(filename):
        molname: str = os.path.splitext(os.path.basename(filename))[0]
        return molname.replace("output_","").upper()


def compare2files(file1, file2):
    """
    Given two molecular orbital json files, print out certain MO values
    for comparison.
    """
    dpi = 80
    fig = plt.figure(figsize=(1600/dpi, 1600/dpi), dpi=dpi)
    file1_name = reducefname(os.path.basename(file1))
    file2_name = reducefname(os.path.basename(file2))
    ### Just comparing HOMO for now
    homo_num, lumo_num = MolecularOrbital.homoLumoNumbersFromJson(file1)
    homo1 = MolecularOrbital.fromJsonFile(file1, homo_num)
    homo1_principle_moments = homo1.get_principle_moments()
    homo1_principle_axes = homo1.get_principle_axes()
    ax1 = homo1.plot(file1_name, 121, fig)

    homo_num, lumo_num = MolecularOrbital.homoLumoNumbersFromJson(file2)
    homo2 = MolecularOrbital.fromJsonFile(file2, homo_num)
    homo2_principle_moments = homo2.get_principle_moments()
    homo2_principle_axes = homo2.get_principle_axes()
    ax2 = homo2.plot(file2_name, 122, fig)


    print(f"{file1_name} principle moments: ")
    print(homo1_principle_moments)
    print(f"{file1_name} principle axes: ")
    print(homo1_principle_axes)
    print(f"{file2_name} principle moments:")
    print(homo2_principle_moments)
    print(f"{file2_name} principle axes: ")
    print(homo2_principle_axes)

    def update(i):
        ax1.view_init(10, i*2)
        ax2.view_init(10, i*2)
        return fig
    
    # if len(input("Show next plot?")) > 0:
    #     plt.show()
    
    ### save fig
    # fig.tight_layout()
    # anim = FuncAnimation(fig, update, frames=180, interval=1)
    # anim.save(f"{file1_name}-vs-{file2_name}.gif", dpi=80, writer="imagemagick")


def main(directory):
    "directory contains the molecular orbitals json files"

    # filepairs = combinations(
    #     [x for x in os.listdir(directory) if x.endswith(".json") and not "ZINC" in x]
    #     , 2)

    filepairs = [
            # ("output_naphthalene.json", "output_butyl_naphthalene.json")
            # , ("output_anthracene.json", "output_butyl_anthracene.json")
            # , ("output_naphthalene.json", "output_naphthalene_butyl_anthracene.json")
            # , ("output_anthracene.json", "output_naphthalene_butyl_anthracene.json")
            # , ("output_diphenyl_butadiene.json", "output_diphenyl_hexatriene.json")
            ("output_new_butyl-naphtalene.json", "output_new_naphtalene.json")
    ]

    for file1, file2 in filepairs:
        print(f"Compare {reducefname(file1)} and {reducefname(file2)}\n".upper())
        naphthalene_file = os.path.join(directory, file1)
        butyl_naphthalene_file = os.path.join(directory, file2)
        compare2files(naphthalene_file, butyl_naphthalene_file)

        print("\n----------------------------------------------\n")
if __name__ == "__main__":
    main(
        os.path.join("..","sampleInputs","PM7_orbitals")
    )