import os
from pprint import pprint
from typing import Mapping, NamedTuple, Union
from itertools import combinations
from y4_python.python_modules.regression import MyRegression

from matplotlib import pyplot as plt

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from .running_orbital_calculations import logfun, main as orbital_calculations_main, reducefname

from .python_modules.database import main as db_main, DB

from .python_modules.orbital_similarity import radial_distribution_difference, sort_molecular_orbital_pairs, orbital_distance
from .python_modules.structural_similarity import structural_distance
from .python_modules.orbital_calculations import MolecularOrbital

from .learning import main as learning_main

print("__init__.py called.")

def orbital_calculations():
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    orbital_calculations_main(orbitalDir)



def print_sorted_orbital_pairs(orbital_distance_kwargs):
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
        homo = MolecularOrbital.fromJsonFile(fname, MolecularOrbital.HOMO, molecule_name=reducefname(fname))
        return homo.toDict()

    orbitals = map(fun, files)
    sortedOrbitals = sort_molecular_orbital_pairs(orbitals, orbital_distance_kwargs)

    
    results = [sorted([x[0]["molecule_name"], x[1]["molecule_name"]]) + [x[2]] for x in sortedOrbitals]
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

def plot_all_radial_dist(radial_distribution_kwargs, outDir=False):
    """
    For each file in files, print out the logfun (molname, homo num, centre of mass, principle moments)
    """
    from .python_modules.orbital_calculations import MolecularOrbital as MO
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    for f in os.listdir(orbitalDir):
        filename = os.path.join(orbitalDir, f)
        homo = MO.fromJsonFile(filename, MO.HOMO, molecule_name=reducefname(f).capitalize(), radial_distribution_kwargs=radial_distribution_kwargs)
        X,F = homo.radial_dist_func(**radial_distribution_kwargs)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(X,F)
        #ax.set_title(homo.molecule_name + f", {radial_distribution_kwargs}")
        ax.set_xlabel(r"$X_{RDF}$ / $\AA$")
        ax.set_ylabel(r"$F_{RDF}$ / $E_h\,^2$")
        print(f"{homo.molecule_name} done. Sum = {sum(F)}")
        if outDir:
            fig.savefig(os.path.join(outDir, homo.molecule_name))
    plt.show()

def check_r_rmse_for_different_kNeighbors():
    import numpy as np
    from scipy.stats import linregress
    from sklearn.metrics import mean_absolute_error
    d = r"results\2021-03-30\11k_molecule_database_eV"
    out = []
    for fold in os.listdir(d):
        folder = os.path.join(d,fold)
        f = os.listdir(folder)[0]
        filepath = os.path.join(folder,f)
        filepath = os.path.join(filepath, os.listdir(filepath)[0])
        results = np.load(filepath).T
        lr = linregress(results[2], results[1])
        rvalue = lr.rvalue
        rmse = mean_absolute_error(results[1], lr.intercept + lr.slope*results[2])
        out.append(f"{fold},{rvalue:.4f},{rmse:.4f}")
    print("n_neigh  rvalue  rmse")
    print("\n".join(out))

#orbital_calculations()

#changing_weight_scaling_factor()

#print_sorted_orbital_pairs()
#print_all_inertia_info()

#plotting()

# db_main()
if __name__ == "__main__":
    ""
    import numpy as np
    from scipy.stats import linregress

    plot_all_radial_dist({})

    # print_sorted_orbital_pairs(
    #     {
    #         "inertia_coeff":0
    #         , "IPR_coeff":0
    #         , "O_coeff":0
    #         , "N_coeff":0
    #         , "S_coeff":0
    #         , "P_coeff":0
    #         , "radial_distribution_coeff": 1
    #     }
    # )

    # db = DB(os.path.join("y4_python","11k_molecule_database_eV.db"))
    # regression = MyRegression(db)
    # all_ = db.get_all()
    # results = np.load(r"results\2021-03-30\11k_molecule_database_eV\n_neigh=3\inertia_distance\inertia_distance.npy")
    # def mfilter(results_row):
    #     idx = int(results_row[0])
    #     row_i = all_[idx]
    #     molid, pm7, blyp, *_ = row_i
    #     dE_i = abs(regression.distance_from_regress(pm7, blyp))
    #     return dE_i > 0.2401 and results_row[2] < 0.5
    # filtered = filter(mfilter, results)
    # filtered = np.array(list(filtered)).T
    # lr = linregress(filtered[2], filtered[1])
    # print(lr)
    # h = plt.hist2d(filtered[2], filtered[1], bins=100, cmin=1)
    # plt.colorbar(h[3])
    # # plt.scatter(filtered[2], filtered[1])
    # plt.xlabel(r"Inertia Distance, $\overline{D}_{n,k}$")
    # plt.ylabel(r"$\overline{Y}_{n,k}$ / eV")
    # plt.show()

    