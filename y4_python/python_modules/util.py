import os
import sqlite3
from typing import Callable, Iterable, List, Optional, Tuple, Union, overload

from rdkit.Chem import Mol, SanitizeFlags, SanitizeMol, MolFromSmiles, RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprint

from rdkit.DataStructs.cDataStructs import BitVectToBinaryText, CreateFromBinaryText, ExplicitBitVect

import numpy as np

from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

def create_dir_if_not_exists(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        return

def sanitize_without_hypervalencies(m: Mol):
    ### Sanitize molecule (without checking for hypervalencies)
    SanitizeMol(
        m
        ,SanitizeFlags.SANITIZE_FINDRADICALS|SanitizeFlags.SANITIZE_KEKULIZE|SanitizeFlags.SANITIZE_SETAROMATICITY|SanitizeFlags.SANITIZE_SETCONJUGATION|SanitizeFlags.SANITIZE_SETHYBRIDIZATION|SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True
        )


def distance_x_label(distance_fun:Callable):
    return ' '.join(
        (x.capitalize() for x in distance_fun.__name__.split("_"))
    )

# Fingerprint constants
def mol_from_smiles(smiles:str) -> Mol:
    m = MolFromSmiles(smiles, sanitize=False)
    sanitize_without_hypervalencies(m)
    return m
class Consts:
    MORGAN_FP = 1
    RDK_FP = 2

def fingerprint_from_smiles(smiles:str, fingerprint_type: int) -> ExplicitBitVect:
    """
    TODO: generalize to multiple fingerprints
    for now Morgan and RDK (daylight) Fingerprint
    """
    funMap = {
        Consts.MORGAN_FP: GetMorganFingerprint,
        Consts.RDK_FP: RDKFingerprint
    }

    m = mol_from_smiles(smiles)
    return funMap[fingerprint_type](m)

def density_scatter( x , y, ax = None, fig=None, sort = True, bins = 20, cmap='jet', **kwargs ):
    """
    Scatter plot colored by 2d histogram

    Taken from here: https://stackoverflow.com/a/53865762/10326441

    """
    x = np.array(x)
    y = np.array(y)

    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = np.int(0.0)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, cmap=cmap, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def scale_array(X: np.ndarray, min_, max_)-> np.ndarray:
    """
    See here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max_ - min_) + min_
    return X_scaled


HARTREE_IN_EV = 27.21138624598853 # https://en.wikipedia.org/wiki/Hartree_atomic_units#Units
@overload
def atomic_units2eV(au: Union[int, float]) -> float:...

@overload
def atomic_units2eV(au: np.ndarray) -> np.ndarray:...

def atomic_units2eV(au: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    return au * HARTREE_IN_EV

def verify_filename(filename:str):
    "Check if filename exists, if exists increment filename with an integer"
    i = 1
    exists = os.path.isfile(filename)
    new_filename, ext = os.path.splitext(filename)
    while exists:
        new_filename = new_filename + f"({i})"
        exists = os.path.isfile(new_filename + ext)
        i += 1
    
    return new_filename + ext


def absolute_mean_deviation_from_y_equals_x(x,y):
    return sum(
        abs(y[idx] - x[idx]) for idx in range(len(x))
    ) / len(x)
    

def plot_medians_iqr(X_data, Y_data, bins: Union[List[float], np.ndarray]) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''

    Example:
        X_data = ...
        Y_data = ...
        data = np.array([X,Y]).T  # [[x1,y1], [x2,y2],...]
        X, Q1, Q2, Q3 = plot_medians_iqr(data)
        plt.plot(X, Q1)
        plt.plot(X, Q2)
        plt.plot(X, Q3)

    Given data in X and Y, we will split into bins along X. 
    Then for each bin, we calculate the Q1 (lower quartile), Q2 (median), and Q3 (upper quartile).
    We then produce a plot, where X is the middle of the bins and Y contains 3 line plots, one for 
    each of Q1, Q2 and Q3.

    As far as I know 'doane' is the best algorithm for bins.
    It might take a while but thats fine.
    The more bins there are the smoother the plot should look, although if the bins are too
    small, then there won't be a significant Y distribution within that range.

    ------------
    Returns:
        X: List (middle of each bin)
        Q1_list:    List of lower quartiles
        Q2_list:    List of medians
        Q3_list:    List if upper quartiles

    '''

    X_out = [] # middle of the bins
    Q1_list = []
    Q2_list = []
    Q3_list = []
    for i in range(len(bins)-1):
        bin = (bins[i], bins[i+1])
        data_slice = Y_data[(bin[0] < X_data) & (X_data < bin[1])] # get a slice of data for our bin range
        ### TODO: handle if the data_slice is empty.
        if not data_slice.any():
            continue
        Q1 = np.percentile(data_slice, 25)
        Q2 = np.percentile(data_slice, 50)
        Q3 = np.percentile(data_slice, 75)
        X_out.append(sum(bin)/len(bin)) 
        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Q3_list.append(Q3)

    return X_out, Q1_list, Q2_list, Q3_list

filenames2smiles = {
    "naphthalene-butyl-anthracene": "C1=CC=CC2=C1C=C3C(=C2)C=CC(=C3)CCCCC4=CC5=C(C=C4)C=CC=C5"
    , "naphthalene": "C1=CC=CC=C1"
    , "diphenyl-butadiene":"C1=CC=CC=C1C=CC=CC2=CC=CC=C2"
    , "diphenyl-hexatriene":"C1=CC=CC=C1C=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-octatetrene":"C1=CC=CC=C1C=CC=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-decapentene":"C1=CC=CC=C1C=CC=CC=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-dodecahexene":"C1=CC=CC=C1C=CC=CC=CC=CC=CC=CC2=CC=CC=C2"
    , "anthracene": "C1=CC=C2C=C3C=CC=CC3=CC2=C1"
    , "butyl-anthracene": "C1=CC=CC2=C1C=C3C(=C2)C=CC(=C3)CCCC"
}