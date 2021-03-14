import os
import sqlite3
from typing import Union, overload

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


# Fingerprint constants
class Consts:
    MORGAN_FP = 1
    RDK_FP = 2

def fingerprint_from_smiles(s, fingerprint_type: int) -> ExplicitBitVect:
    """
    TODO: generalize to multiple fingerprints
    for now Morgan and RDK (daylight) Fingerprint
    """
    funMap = {
        Consts.MORGAN_FP: GetMorganFingerprint,
        Consts.RDK_FP: RDKFingerprint
    }

    m = MolFromSmiles(s, sanitize=False)
    sanitize_without_hypervalencies(m)
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




filenames2smiles = {
    "naphthalene-butyl-anthracene": "C1=CC=CC2=C1C=C3C(=C2)C=CC(=C3)CCCCC4=CC5=C(C=C4)C=CC=C5"
    , "naphthalene": "C1=CC=CC=C1"
    , "diphenyl-butadiene":"C1=CC=CC=C1C=CC=CC2=CC=CC=C2"
    , "diphenyl-hexatriene":"C1=CC=CC=C1C=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-octatetrene":"C1=CC=CC=C1C=CC=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-decapentene":"C1=CC=CC=C1C=CC=CC=CC=CC=CC2=CC=CC=C2"
    , "diphenyl-dodecahexene":"C1=CC=CC=C1C=CC=CC=CC=CC=CC=CC2=CC=CC=C2"
}