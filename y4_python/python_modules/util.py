import os

from rdkit.Chem import Mol, SanitizeFlags, SanitizeMol, MolFromSmiles, RDKFingerprint

from rdkit.DataStructs.cDataStructs import BitVectToBinaryText, CreateFromBinaryText

import numpy as np

from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

def create_dir_if_not_exists(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        return

def sanitize_without_hypervalencies(m: Mol):
    ### Sanitize molecule (without checking for hypervalencies)
    SanitizeMol(
        m
        ,SanitizeFlags.SANITIZE_FINDRADICALS|SanitizeFlags.SANITIZE_KEKULIZE|SanitizeFlags.SANITIZE_SETAROMATICITY|SanitizeFlags.SANITIZE_SETCONJUGATION|SanitizeFlags.SANITIZE_SETHYBRIDIZATION|SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True
        )

def fingerprint_from_smiles(s):
    m = MolFromSmiles(s, sanitize=False)
    sanitize_without_hypervalencies(m)
    return RDKFingerprint(m)

def density_scatter( x , y, ax = None, sort = True, bins = 20, cmap='jet', **kwargs )   :
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
