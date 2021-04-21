from typing import Callable, Iterable, List, Tuple, Union
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
from sklearn.metrics import mean_absolute_error

from matplotlib import pyplot as plt


from scipy.stats import pearsonr, linregress

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from typing_extensions import TypedDict

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from .python_modules.util import absolute_mean_deviation_from_y_equals_x, sanitize_without_hypervalencies, create_dir_if_not_exists, verify_filename
from .python_modules.regression import MyRegression
from .python_modules.orbital_calculations import SerializedMolecularOrbital
from .python_modules.structural_similarity import structural_distance
from .python_modules.orbital_similarity import orbital_distance
from .python_modules.chemical_distance_metric import chemical_distance, MetricParams
from .python_modules.database import DB

def plot(x, y, data_label, x_label, y_label,):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y, label=data_label)
    ax.plot(x,x, color='red', label = "y=x")
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    plt.tight_layout()
    return fig,ax

def hist(x, y, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot()
    h = ax.hist2d(x, y, bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    plt.tight_layout()
    return fig,ax
    

def knn(k_neighbors, k_folds, X, y, distance_fun, metric_params, weights='distance'):

    if k_folds == -1:
        "Leave-One-Out"
        k_folds = len(y)

    y_predicted = []
    y_real = []

    # n_jobs = -1, means use all CPUs
    knn = neighbors.KNeighborsRegressor(k_neighbors, weights=weights, metric=distance_fun, metric_params=metric_params, n_jobs=1) 

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
    rmse = mean_absolute_error(y_real, y_predicted)
    return (r, rmse)

def main_chemical_distance(
        k_neighbours: int
        , k_folds: int
        , metric_params: MetricParams
        , mol_list # our X data
        , deviation_list # our y data
        , weights='distance'
        , save=True
    ):
    """
    X = smiles_list
    y = deviation from regress line

    """ 


    X = np.asarray([i for i in range(len(mol_list))]) # input data -> index for each row of the database IE each molecule
    y = np.asarray(deviation_list) # expected output eg regression deviation 

    # Sampling for testing TODO: COMMENT OUT
    # cutoff = 10
    # X = X[:cutoff]
    # y = y[:cutoff]

    verbose_results = []
    training_start_time = datetime.today()
    start = perf_counter()
    y_real, y_predicted, r, rmse = knn(k_neighbors=k_neighbours, k_folds=k_folds, X=X, y=y, distance_fun=chemical_distance, metric_params=metric_params, weights=weights)
    verbose_results = y_real, y_predicted, r, rmse, k_neighbours, k_folds, metric_params, training_start_time
    finish = perf_counter()
    if save:
        save_results(*verbose_results)
    
    print(f"time taken to train = {round(finish - start, ndigits=5)} seconds.")

    return y_real, y_predicted, r, rmse

def save_results(y_real:np.ndarray, y_predicted:np.ndarray, r, rmse, k_neighbors:int, k_folds:int, params:MetricParams, training_start_time: date):
    """
    Export y_real,y_predicted in csv format.
    FileName will be "learning" + the start time of the training.
    We will also output a info file, listing the parameters.
    """

    out_folder = str(date.today())
    filename = training_start_time.strftime("%H-%M-%S")
    fpath = os.path.join("Results", out_folder, "learning_" + filename)
    
    create_dir_if_not_exists(os.path.dirname(fpath))

    ### Don't forget to add extension
    info_file_path = verify_filename(fpath + "-info.txt")
    with open(info_file_path, "w") as InfoFile:
        InfoFile.write(f"r={r}\nrmse={rmse}\nk_neighbors={k_neighbors}\nk_folds={k_folds}\n")
        InfoFile.writelines([f"{key}={val}\n" for key,val in params.items() if "list" not in key])

    npy_fpath = verify_filename(fpath + ".npy")

    np.save(npy_fpath, np.array((y_real, y_predicted)).T)


def show_results(results_file):
    "plot the results from a given file."
    y_real, y_predicted = results = np.load(results_file).T

    reg = linregress(y_real, y_predicted)
    x_label=r'$y_{real} \, / \, eV$'
    y_label=r'$y_{pred} \, / \, eV$'
    fig1, ax1 = plot(x=y_real, y=y_predicted, data_label="predicted vs real, k=5", x_label=x_label, y_label=y_label)
    ax1.plot(y_real, reg.slope*y_real+reg.intercept, color=(222/255,129/255,29/255), label="linear regression")
    ax1.legend()
    fig2, ax2 = hist(x=y_real, y=y_predicted, x_label=x_label, y_label=y_label)
    ax2.plot(y_real, reg.slope*y_real+reg.intercept, color=(222/255,129/255,29/255), label="linear regression")
    ax2.plot(y_real,y_real, color='red', label = "y=x")
    ax2.legend()

    plt.show()


def main(db: DB, metric_params:MetricParams):

    my_regression = MyRegression(db)
    mol_list = db.get_mol_ids()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    deviation_list = (list(map(my_regression.distance_from_regress, pm7_energies, blyp_energies)))
    fingerprint_list = db.get_fingerprints()
    homo_molecular_orbital_list = db.get_homo_molecular_orbitals()
    lumo_molecular_orbital_list = db.get_lumo_molecular_orbitals()

    metric_params["fingerprint_list"] = fingerprint_list
    metric_params["homo_orbital_list"] = homo_molecular_orbital_list
    metric_params["lumo_orbital_list"] = lumo_molecular_orbital_list

    return main_chemical_distance(
        k_neighbours=5
        , k_folds=10
        , metric_params=metric_params
        , mol_list=mol_list
        , deviation_list=deviation_list
    )

if __name__ == "__main__":
    import sys
    import json
    db_path = sys.argv[1]
    params_file = sys.argv[2]

    ### ParamsFile should be object of type MetricParams

    with open(params_file, 'r') as ParamsFile:
        params = json.load(ParamsFile)
    db = DB(db_path)
    y_real, y_pred, r, rmse = main(db, params)
    print(f"r={r:.4f}, rmse={rmse:.4f}")