import os
from pprint import pprint

from .running_orbital_calculations import main, reducefname

def orbital_calculations():
    orbitalDir = os.path.join("sampleInputs", "PM7_optimisedOrbitals")
    main(orbitalDir)

#orbital_calculations()

from .python_modules.orbital_similarity import CalculatedMolecularOrbital, sort_molecular_orbital_pairs
from .python_modules.orbital_calculations import MolecularOrbital

def run2():
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
    

    
run2()


def run3():
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


#run3()