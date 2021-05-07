# molecular-descriptors

Some code relevant to computing descriptors for calculated molecular orbitals, including:
    - Orbital Moment of Inertia
    - Percentage Orbital Weighting on Heteroatom
    - Radial Distribution Function

As well as distance functions computing the distance between two molecules based on the above descriptors.

And Simple k-NN regression with sklearn.

Any questions or issues can be discussed via the GitHub issues or discussion sections.

Here is a table of some files and a brief description of each. Please also check the 
Documentation inside each file.
Brief description of contents:

| File                  | Description   |
|:-------------:        |:-------------:|
| parse_orbitalsV2.bat  | Parse a gaussian log file and produce .json file with atomic coordinates and orbital coefficients|
| y4_python/python_modules/orbital_calculations.py | Includes the MolecularOrbital class which involves molecular orbital calculations. Create a class with MolecularOrbital.fromJsonFile(..) method. Useful properties (e.g. inertia tensor, RDF) can be accessed as attributes.|
| y4_python/python_modules/database.py | Database API, for storing useful information related to a dataset of molceules. See the main function for creating database. DB class methods can be used for reading the database. |
| y4_python/python_modules/chemical_distance_metric.py | Defines a Chemical distance metric function for use in k-NN algorithm, which combines structural and orbital distances. |
| y4_python/python_modules/orbital_similarity.py | Defines a orbital distance functions. |
| y4_python/python_modules/structural_similarity.py | Defines a structural distance function based on molecular fingerprints. |
| y4_python/python_modules/descriptors.py | For calculating some descriptors for molecules. |
