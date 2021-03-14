from typing import Callable, List, Tuple
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

from .python_modules.util import sanitize_without_hypervalencies, create_dir_if_not_exists
from .python_modules.regression import distance_from_regress
from .python_modules.orbital_calculations import SerializedMolecularOrbital
from .python_modules.structural_similarity import structural_distance
from .python_modules.orbital_similarity import orbital_distance
from .python_modules.chemical_distance_metric import chemical_distance, MetricParams
from .python_modules.database import DB

def pre_calculated(i, j, calculated_pairs: dict):
    ""

def plot(x, y, data_label, x_label, y_label, title=None,):
    if title == None:
        title = "k-NN With Chemical Distance Metric"
    ax = plt.subplot()
    ax.scatter(x, y, label=data_label)
    ax.plot(x,x, color='green', label = "y=x")
    ax.legend()
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    #ax.set_title(title)
    plt.tight_layout()

def knn(k_neighbors, k_folds, X, y, metric_params, weights='distance'):
    y_predicted = []
    y_real = []

    # n_jobs = -1, means use all CPUs
    knn = neighbors.KNeighborsRegressor(k_neighbors, weights=weights, metric=chemical_distance, metric_params=metric_params, n_jobs=-1) 

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

def main_euclidean():
    db = DB()

    pm7_energies_with_smiles = np.asarray(db.get_pm7_energies())
    blyp_energies = np.asarray(db.get_blyp_energies())

    ### Try training without chemicalDistance

    X = pm7_energies_with_smiles # input data eg pm7 energies, 
    y = blyp_energies # expected output eg blyp energies

    results = [("k", "r", "rmse")]
    for k in range(1,31):
        results.append((k,) + knn(k_neighbors=k, X=X, y=y))

    for line in results:
        print(",".join(str(x) for x in line))

def main_chemical_distance(k_neighbours, k_folds, orbital_coefficients: List[float]=[0.0,1.0], structural_coefficients:List[float]=[0.0, 1.0], weights='distance'):
    """
    X = smiles_list
    y = deviation from regress line

    """ 

    db = DB()
    ### (mol_name, E_pm7, E_bly, SMILES, rdk_fingerprint, serialized_molecular_orbital)...

    mol_list = db.get_mol_ids()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    smiles_list = db.get_smiles()
    global fingerprint_list
    fingerprint_list = db.get_fingerprints()
    molecular_orbital_list = db.get_molecular_orbitals()

    ### deviation_list = [[regression_error_value1, mol_name1],...]
    deviation_list = (list(map(distance_from_regress, pm7_energies, blyp_energies)))

    #BLYP_minus_PM7_list = [blyp_energies[idx] - pm7_energies[idx] for idx in range(len(blyp_energies))]
    X = np.asarray([i for i in range(len(mol_list))]) # input data - index for each row of the database IE each molecule
    y = np.asarray(deviation_list) # expected output eg regression deviation 

    ### Sampling for testing
    cutoff = None
    X = X[:cutoff]
    y = y[:cutoff]
    pm7_energies = pm7_energies[:cutoff]
    blyp_energies = blyp_energies[:cutoff]


    metric_params: MetricParams = {
        "fingerprint_list": fingerprint_list
        , "molecular_orbital_list": molecular_orbital_list
        , "c_orbital":1.0
        , "c_struct":1.0
        , "inertia_coefficient":1.0
        , "IPR_coefficient":1.0
        , "N_coefficient":1.0
        , "O_coefficient":1.0
    }
    
    
    results: List[Tuple] = [("k_neighbors", "c_orbital", "c_struct", "r", "rmse")]
    verbose_results = []
    training_start_time = datetime.today()
    start = perf_counter()
    for c_orbital in orbital_coefficients:
        for c_struct in structural_coefficients:
            metric_params["c_orbital"] = c_orbital
            metric_params["c_struct"] = c_struct
            y_real, y_predicted, r, rmse = knn(k_neighbors=k_neighbours, k_folds=k_folds, X=X, y=y, metric_params=metric_params, weights=weights)
            verbose_results.append((y_real, y_predicted, r, rmse, k_neighbours, k_folds, metric_params, training_start_time))
            results.append((k_neighbours, c_orbital, c_struct, r, rmse))
    finish = perf_counter()
    for row in verbose_results: save_results(*row)
    # plot(x=y_real, y=y_predicted, data_label="predicted vs real, k=5", x_label=r'$y_{real}$', y_label=r'$y_{pred}$')
    # r, rmse = get_r_rmse(y_real, y_predicted)
    # print(f"r={r} rmse={rmse}")
    # plt.show()
    # plot(x=pm7_energies, y=blyp_energies, data_label="pm7 vs blyp", x_label="PM7 Energy (AU)", y_label="BLYP Energy (AU)")
    # r, rmse = get_r_rmse(pm7_energies, blyp_energies)
    # print(f"r={r} rmse={rmse}")
    # plt.show()
    # blyp_predicted = [pm7_energies[idx] + y_predicted[idx] for idx in range(len(pm7_energies))]
    # plot(x=blyp_energies, y=blyp_predicted, data_label="predicted blyp vs blyp", x_label="BLYP Energy (AU)", y_label="Predicted BLYP Energy (PM7 Energy + Correction) ")
    # r, rmse = get_r_rmse(blyp_energies, blyp_predicted)
    # print(f"r={r} rmse={rmse}")
    # plt.show()
    print(f"time taken to train = {round(finish - start, ndigits=5)}")
    for line in results:
        print(",".join(str(x) for x in line))

def save_results(y_real:np.ndarray, y_predicted:np.ndarray, r, rmse, k_neighbors:int, k_folds:int, params:MetricParams, training_start_time: date):
    """
    Export y_real,y_predicted in csv format.
    FileName will be the start time of the training.
    We will also output a info file, listing the parameters.
    """

    out_folder = str(date.today())
    filename = training_start_time.strftime("%H-%M-%S")
    fpath = os.path.join("Results", out_folder, filename)
    create_dir_if_not_exists(os.path.dirname(fpath))

    ### Don't forget to add extension
    with open(fpath + "-info.txt", "w") as InfoFile:
        InfoFile.write(f"r={r}\nrmse={rmse}")
        InfoFile.writelines([f"{key}={val}" for key,val in params.items() if "list" not in key])
        # infolines = []
        # for key,val in params.items():
        #     if "line" in key:
        #         continue
        #     else:
        #         infolines.append(
        #             f"{key}={val}"
        #         )
    
    with open(fpath + ".csv", "w", newline='') as CsvFile:
        writer = csv.writer(CsvFile)
        writer.writerows(np.array((y_real, y_predicted)).T)


def show_results(results_file):
    "plot the results from a given file."

def main():
    main_chemical_distance(k_neighbours=5, k_folds=10, orbital_coefficients=[0.5], structural_coefficients=[1])
    main_chemical_distance(k_neighbours=5, k_folds=10, orbital_coefficients=[0.5], structural_coefficients=[1], weights='uniform')

if __name__ == "__main__":
    main()