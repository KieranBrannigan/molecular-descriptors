from typing import Callable, Iterable, List, Tuple
import sqlite3
import math
from time import perf_counter
from dataclasses import dataclass
import os
import csv
from datetime import date, datetime


from random import sample

from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt


from scipy.stats import pearsonr, linregress

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from typing_extensions import TypedDict

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from .python_modules.util import sanitize_without_hypervalencies, create_dir_if_not_exists, verify_filename
from .python_modules.regression import MyRegression
from .python_modules.orbital_calculations import SerializedMolecularOrbital
from .python_modules.structural_similarity import structural_distance
from .python_modules.orbital_similarity import orbital_distance
from .python_modules.chemical_distance_metric import chemical_distance, MetricParams
from .python_modules.database import DB

def plot(x, y, data_label, x_label, y_label,):
    ax = plt.subplot()
    ax.scatter(x, y, label=data_label)
    ax.plot(x,x, color='green', label = "y=x")
    ax.legend()
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    plt.tight_layout()

def knn(k_neighbors, k_folds, X, y, metric_params, weights='distance'):

    if k_folds == -1:
        "Leave-One-Out"
        k_folds = len(y)

    y_predicted = []
    y_real = []

    # n_jobs = -1, means use all CPUs
    knn = neighbors.KNeighborsRegressor(k_neighbors, weights=weights, metric=chemical_distance, metric_params=metric_params, n_jobs=1) 

    kf = KFold(n_splits=k_folds, shuffle=True)

    for train_index, test_index in kf.split(X):
        # Assign train/test values
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        # Predict data
        y_pred = knn.fit(X_train, y_train.ravel()).predict(X_test)
        # Append data of each leave-one-out iteration
        y_predicted.append(y_pred.tolist())
        y_real.append(y_test.tolist())

    ### Flatten lists with real and predicted values
    y_real = [item for dummy in y_real for item in dummy ]
    y_predicted = [item for dummy in y_predicted for item in dummy ]
    ### Calculate r and rmse metrics
    r, rmse = get_r_rmse(y_real, y_predicted)
    return (y_real, y_predicted, r, rmse)

def get_r_rmse(y_real, y_predicted):
    r, _ = pearsonr(y_real, y_predicted)
    rmse = math.sqrt(mean_squared_error(y_real, y_predicted))
    return (r, rmse)

def main_chemical_distance(db: DB, k_neighbours, k_folds, metric_params_list: Iterable[MetricParams], weights='distance'):
    """
    X = smiles_list
    y = deviation from regress line

    """ 

    mol_list = db.get_mol_ids()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    smiles_list = db.get_smiles()
    fingerprint_list = db.get_fingerprints()
    homo_molecular_orbital_list = db.get_homo_molecular_orbitals()

    my_regression = MyRegression(db)

    deviation_list = (list(map(my_regression.distance_from_regress, pm7_energies, blyp_energies)))

    X = np.asarray([i for i in range(len(mol_list))]) # input data -> index for each row of the database IE each molecule
    y = np.asarray(deviation_list) # expected output eg regression deviation 

    ## Sampling for testing
    cutoff = None
    X = X[:cutoff]
    y = y[:cutoff]
    pm7_energies = pm7_energies[:cutoff]
    blyp_energies = blyp_energies[:cutoff]

    verbose_results = []
    training_start_time = datetime.today()
    start = perf_counter()
    for metric_params in metric_params_list:
        metric_params["fingerprint_list"] = fingerprint_list
        metric_params["molecular_orbital_list"] = homo_molecular_orbital_list
        y_real, y_predicted, r, rmse = knn(k_neighbors=k_neighbours, k_folds=k_folds, X=X, y=y, metric_params=metric_params, weights=weights)
        verbose_results.append((y_real, y_predicted, r, rmse, k_neighbours, k_folds, metric_params, training_start_time))
    finish = perf_counter()
    for row in verbose_results: save_results(*row)
    
    print(f"time taken to train = {round(finish - start, ndigits=5)}")

def save_results(y_real:np.ndarray, y_predicted:np.ndarray, r, rmse, k_neighbors:int, k_folds:int, params:MetricParams, training_start_time: date):
    """
    Export y_real,y_predicted in csv format.
    FileName will be the start time of the training.
    We will also output a info file, listing the parameters.
    """

    out_folder = str(date.today())
    filename = training_start_time.strftime("%H-%M-%S")
    fpath = os.path.join("Results", out_folder, "learning_" + filename)
    
    create_dir_if_not_exists(os.path.dirname(fpath))

    ### Don't forget to add extension
    info_file_path = verify_filename(fpath + "-info.txt")
    with open(info_file_path, "w") as InfoFile:
        InfoFile.write(f"r={r}\nrmse={rmse}")
        InfoFile.writelines([f"{key}={val}\n" for key,val in params.items() if "list" not in key])

    npy_fpath = verify_filename(fpath + ".npy")

    np.save(npy_fpath, np.array((y_real, y_predicted)).T)


def show_results(results_file):
    "plot the results from a given file."
    y_real, y_predicted = results = np.load(results_file)
    plot(x=y_real, y=y_predicted, data_label="predicted vs real, k=5", x_label=r'$y_{real} / eV$', y_label=r'$y_{pred} / eV$')
    plt.show()


def main(db: DB):
    metric_params_list: List[MetricParams] = [
        {
            "fingerprint_list":[]
            , "molecular_orbital_list":[]
            , "c_orbital": 1.0
            , "inertia_coefficient": 1.0
            , "IPR_coefficient": 0
            , "N_coefficient": 0
            , "O_coefficient": 0
            , "S_coefficient": 0
            , "P_coefficient": 0
            , "c_struct": 0
        }
        , {
            "fingerprint_list":[]
            , "molecular_orbital_list":[]
            , "c_orbital": 0.0
            , "inertia_coefficient": 1.0
            , "IPR_coefficient": 0
            , "N_coefficient": 0
            , "O_coefficient": 0
            , "S_coefficient": 0
            , "P_coefficient": 0
            , "c_struct": 1
        }
    ]
    main_chemical_distance(db=db, k_neighbours=5, k_folds=-1, metric_params_list=metric_params_list)

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1]
    db = DB(db_path)
    main(db)