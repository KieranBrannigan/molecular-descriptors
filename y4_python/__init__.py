import os
from pprint import pprint

from matplotlib import pyplot as plt

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from .running_orbital_calculations import logfun, main as orbital_calculations_main, reducefname

from .python_modules.database import main as db_main, DB

from .python_modules.orbital_similarity import sort_molecular_orbital_pairs
from .python_modules.orbital_calculations import MolecularOrbital

from .learning import main as learning_main

def orbital_calculations():
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    orbital_calculations_main(orbitalDir)



def print_sorted_orbital_pairs():
    """
    Gets the homo of each file in files.
    Compares every pair based on distance.
    Prints the sorted pairs in order of increasing distance
    """
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    
    files = set((f for f in os.listdir(orbitalDir)))
    def fun(x):
        # x is file
        fname = os.path.join(orbitalDir, x)
        homo = MolecularOrbital.fromJsonFile(fname, MolecularOrbital.HOMO)
        return CalculatedMolecularOrbital(
            inertiaTensor=homo.inertia_tensor
            , principleAxes=homo.principle_axes
            , principleMoments=homo.principle_moments
            , IPR=0.1
            , molecule_name=reducefname(x).capitalize()
        )

    orbitals = map(fun, files)
    sortedOrbitals = sort_molecular_orbital_pairs(orbitals)

    
    results = [(x[0].molecule_name, x[1].molecule_name, x[2]) for x in sortedOrbitals]
    for row in results:
        print(
            ",".join([str(x).replace("_","-").capitalize() for x in row])
            )
    

    


def run3():
    """
    Prints out x_i, y_i, z_i, W_i for homo of each file in files
    """
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    files = {
        'anthracene_output.json',
        'butyl_anthracene_output.json',
        'naphthalene_output.json',
        'butyl_naphthalene_output.json',
        'naphthalene_butyl_anthracene_output.json',
        'diphenyl_butadiene_output.json',
        'diphenyl_hexatriene_output.json',
    }
    def fun(x):
        # x is file
        fname = os.path.join(orbitalDir, x)
        return MolecularOrbital.fromJsonFile(fname, MolecularOrbital.HOMO, molecule_name=reducefname(x).capitalize())
        

    orbitals = map(fun, files)

    out_dir = "orbital_coords_weights_output"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for mo in orbitals:
        #xs, ys, zs, weights, scaledWeights, colours, atom_numbers = mo.get_atom_plot_values()

        rows = []

        for atom_number, atomic_contribution in mo.mo["atomic_contributions"].items():
            x,y,z = mo.atomic_coords[atom_number]
            atom_weight = mo.calc_atomic_weight(atomic_contribution)
            atom_symbol = atomic_contribution['atom_symbol']
            rows.append(
                (str(i) for i in (x,y,z, atom_weight))
            )
        with open(os.path.join(out_dir, mo.molecule_name) + ".log", "w") as OutFile:
            OutFile.writelines([",".join(row)+"\n" for row in rows])
            # for row in rows:
            #     OutFile.write(",".join(row) + "\n")

def plotting():
    from .plotting import main
    main()


def changing_weight_scaling_factor():
    factors = [1, 2, 3, 5, 10, 15]

    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    test_file = "butyl_naphthalene_output.json"
    #test_file = "anthracene_output.json"

    fig=plt.figure()
    for idx,factor in enumerate(factors):
        homo = MolecularOrbital.fromJsonFile(
            os.path.join(orbitalDir, test_file)
            , mo_number=MolecularOrbital.HOMO
            , molecule_name=reducefname(test_file).capitalize()
            , weight_scaling_factor=factor
        )

        logfun(homo)
        axis_number = int("23" + str(idx+1))
        homo.plot(f"{homo.molecule_name}, weight_scaling_factor = {factor}", axis_number=axis_number, fig=fig)
    
    plt.show()

def print_all_inertia_info():
    """
    For each file in files, print out the logfun (molname, homo num, centre of mass, principle moments)
    """
    from .python_modules.orbital_calculations import MolecularOrbital as MO
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    for f in os.listdir(orbitalDir):
        filename = os.path.join(orbitalDir, f)
        homo = MO.fromJsonFile(filename, MO.HOMO, molecule_name=reducefname(f).capitalize())
        logfun(homo)
        print("----------------------")

#orbital_calculations()

#changing_weight_scaling_factor()

#print_sorted_orbital_pairs()
#print_all_inertia_info()

#plotting()

# db_main()

learning_main()