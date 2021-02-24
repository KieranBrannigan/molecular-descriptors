import os
from pprint import pprint

from .running_orbital_calculations import main, reducefname

def orbital_calculations():
    orbitalDir = os.path.join("sampleInputs", "PM7_orbitals")
    main(orbitalDir)

#orbital_calculations()

from .python_modules.orbital_similarity import CalculatedMolecularOrbital, sort_molecular_orbital_pairs
from .python_modules.orbital_calculations import MolecularOrbital

def run2():
    files = {
        'output_anthracene.json',
        'output_butyl_anthracene.json',
        'output_butyl_naphthalene.json',
        'output_diphenyl_butadiene.json',
        'output_diphenyl_hexatriene.json',
        'output_naphthalene.json',
        'output_naphthalene_butyl_anthracene.json',
        'output_new_butyl-naphtalene.json',
        'output_new_naphtalene.json'
    }
    orbitalDir = os.path.join("sampleInputs", "PM7_orbitals")
    def fun(x):
        # x is file
        fname = os.path.join(orbitalDir, x)
        homo_num, _ = MolecularOrbital.homoLumoNumbersFromJson(fname)
        homo = MolecularOrbital.fromJsonFile(fname, homo_num)
        return CalculatedMolecularOrbital(
            inertiaTensor=homo.inertia_tensor
            , principleAxes=homo.principle_axes
            , principleMoments=homo.principle_moments
            , IPR=0.1
            , molecule_name=reducefname(x).capitalize()
        )

    orbitals = map(fun, files)
    sortedOrbitals = sort_molecular_orbital_pairs(orbitals)

    pprint(
        [(x[0].molecule_name, x[1].molecule_name, x[2]) for x in sortedOrbitals]
    )

    
run2()
