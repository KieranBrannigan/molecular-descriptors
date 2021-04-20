"""

Using a genetic algorithm to optimise a set of parameters for the machine learning model.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html 

"""

from y4_python.python_modules.regression import MyRegression
from scipy.sparse.csr import csr_matrix
from y4_python.python_modules.chemical_distance_metric import MetricParams
from y4_python.python_modules.database import DB
from scipy.optimize import differential_evolution

from .learning import main_chemical_distance as learning_main

def optimise_func(hyperparams, k_neighbors:int, k_folds:int, fixed_hyperparams:MetricParams, mol_list, deviation_list):
    '''
    Assigns hyperparameters, generate ML objects and call appropriate 
    training/testing functions

    Parameters
    ----------
    hyperparams: list
        list containing hyperparameters to be optimized
    fixed_hyperparams: list
        list containing hyperparameters not optimized

    Returns
    -------
    rmse: float
        value of the error metric
    '''

    c_struct, c_orbital = hyperparams

    params = fixed_hyperparams.copy()

    params['c_orbital'] = c_orbital
    params['c_struct'] = c_struct

    y_real, y_pred, r, rmse = learning_main(k_neighbors, k_folds, params, mol_list, deviation_list, save=False)
    return rmse

if __name__ == "__main__":
    c_struct_lim  = (0.0, 10.0)
    c_orbital_lim  = (0.0, 10.0)
    bounds = [c_struct_lim] + [c_orbital_lim] # this will be whatever hyperparameters you want to optimize. The length of this sequence is what indicates the number of hyperparameters mini_args = (X, y, condition,fixed_hyperparams) # this will be a tuple with the rest of arguments needed for your function NCPU = 20 # this will be the number of CPUs you want
    
    num_CPU = 20
    
    db = DB("y4_python/11k_molecule_database_eV.db")
    my_regression = MyRegression(db)
    mol_list = db.get_mol_ids()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    deviation_list = (list(map(my_regression.distance_from_regress, pm7_energies, blyp_energies)))
    fingerprint_list = db.get_fingerprints()
    homo_molecular_orbital_list = db.get_homo_molecular_orbitals()
    lumo_molecular_orbital_list = db.get_lumo_molecular_orbitals()

    fixed_hyperparams = MetricParams(
        homo_coeff=1.0
        , lumo_coeff=0.0
        , fingerprint_list=fingerprint_list
        , homo_orbital_list=homo_molecular_orbital_list
        , lumo_orbital_list=lumo_molecular_orbital_list
        , c_struct=0.0 # TO BE OPTIMISED...
        , c_orbital=0.0 # TO BE OPTIMISED...
        , inertia_coefficient=0.0
        , IPR_coefficient=0.0
        , N_coefficient=0.0
        , O_coefficient=0.0
        , S_coefficient=0.0
        , P_coefficient=0.0
        , radial_distribution_coeff=1.0
    )


    k_neighbors = 5
    k_folds = 10

    mini_args = (
        k_neighbors
        , k_folds
        , fixed_hyperparams
        , mol_list
        , deviation_list
    )

    solver = differential_evolution(optimise_func,bounds=bounds,args=mini_args,popsize=10,tol=0.1,polish=False,workers=2,updating='deferred',seed=None)

    # print best hyperparams
    best_hyperparams = solver.x
    best_rmse = solver.fun
    print('#######################################')
    print('Best hyperparameters:',best_hyperparams)
    print('Best rmse:', best_rmse)
    print('#######################################')