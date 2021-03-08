from typing import Callable
import sqlite3
import math
from time import perf_counter

from sklearn import neighbors
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt


from scipy.stats import pearsonr, linregress

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from .python_modules.database import DB
from .python_modules.util import sanitize_without_hypervalencies
from .python_modules.regression import distance_from_regress

def tanimotoSimilarity(i_fp, j_fp):
    """
    Input i_fp, j_fp  Fingerprints for i and j
    Return the tanimoto similarity between two molecules i and j
    """
    return DataStructs.FingerprintSimilarity(i_fp, j_fp, metric=DataStructs.TanimotoSimilarity)


def chemicalDistance(i: np.ndarray, j: np.ndarray):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    A function which takes two one-dimensional numpy arrays, and returns a distance. 
    
    Returns Distance between molecule i and j (D_ij) such that:
    D_ij = 1 - T(i,j)  where T(i,j) is the tanimoto similarity between molecule i and j.

    Note that in order to be used within the BallTree, the distance must be a true metric. 
    i.e. it must satisfy the following properties:
    Non-negativity: d(x, y) >= 0
    Identity: d(x, y) = 0 if and only if x == y
    Symmetry: d(x, y) = d(y, x)
    Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
    
    """
    global fingerprint_list
    # i_smiles = i[1]
    # j_smiles = j[1]
    i_fp= fingerprint_list[int(i[0])]
    j_fp = fingerprint_list[int(j[0])]
    #print(f"i_smiles = {i_smiles} j_smiles = {j_smiles}")

    T_ij = tanimotoSimilarity(i_fp, j_fp)
    return 1 - T_ij


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

def knnLOO(n_neighbors, X, y):
    y_predicted = []
    y_real = []

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform', metric=chemicalDistance)

    #kf = KFold(n_splits=2, shuffle=True)

    for train_index, test_index in LeaveOneOut().split(X):
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
        results.append((k,) + knnLOO(n_neighbors=k, X=X, y=y))

    for line in results:
        print(",".join(str(x) for x in line))

def main_chemical_distance():
    """
    X = smiles_list
    y = deviation from regress line

    """

    db = DB()
    ### (mol_name, E_pm7, E_bly, SMILES, fingerprint)...

    mol_list = db.get_mol_names()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    smiles_list = db.get_smiles()
    global fingerprint_list
    fingerprint_list = db.get_fingerprints()
    slope, intercept, r_value, p_value, std_err = linregress(pm7_energies,blyp_energies)
    ### deviation_list = [[regression_error_value1, mol_name1],...]
    deviation_list = (list(map(distance_from_regress, pm7_energies, blyp_energies)))

    #BLYP_minus_PM7_list = [blyp_energies[idx] - pm7_energies[idx] for idx in range(len(blyp_energies))]
    X = np.asarray([i for i in range(len(mol_list))]) # input data eg pm7 energies
    y = np.asarray(deviation_list) # expected output eg regression deviation

    results = [("k", "r", "rmse")]
    start = perf_counter()
    for k in range(5,6):
        y_real, y_predicted, r, rmse = knnLOO(n_neighbors=k, X=X, y=y)
        results.append((k,r,rmse))
    finish = perf_counter()
    plot(x=y_real, y=y_predicted, data_label="predicted vs real, k=5", x_label=r'$y_{real}$', y_label=r'$y_{pred}$')
    r, rmse = get_r_rmse(y_real, y_predicted)
    print(f"r={r} rmse={rmse}")
    plt.show()
    plot(x=pm7_energies, y=blyp_energies, data_label="pm7 vs blyp", x_label="PM7 Energy (AU)", y_label="BLYP Energy (AU)")
    r, rmse = get_r_rmse(pm7_energies, blyp_energies)
    print(f"r={r} rmse={rmse}")
    plt.show()
    blyp_predicted = [pm7_energies[idx] + y_predicted[idx] for idx in range(len(pm7_energies))]
    plot(x=blyp_energies, y=blyp_predicted, data_label="predicted blyp vs blyp", x_label="BLYP Energy (AU)", y_label="Predicted BLYP Energy (PM7 Energy + Correction) ")
    r, rmse = get_r_rmse(blyp_energies, blyp_predicted)
    print(f"r={r} rmse={rmse}")
    plt.show()
    print(f"time taken to train = {round(finish - start, ndigits=5)}")
    for line in results:
        print(",".join(str(x) for x in line))

if __name__ == "__main__":
    fingerprint_list = []
    main_chemical_distance()