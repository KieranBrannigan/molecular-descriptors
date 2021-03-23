from .python_modules.database import DB, main


import sys
### Pass the database and input files as arguments
# database_path, orbitalsDir, BLYP_energies_file, PM7_energies_file, SMILES_file = sys.argv[1:]
print(sys.argv)
main(*sys.argv[1:])