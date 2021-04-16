import csv
import os
from os.path import join
import itertools
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union
from datetime import datetime
from y4_python.python_modules.orbital_calculations import SerializedMolecularOrbital

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import numpy as np
from scipy.stats import linregress

from sklearn.neighbors import NearestNeighbors

from matplotlib import pyplot as plt
import seaborn as sns

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from .python_modules.draw_molecule import SMILEStoFiles, concat_images, draw_grid_images, resize_image
from .python_modules.database import DB
from .python_modules.regression import MyRegression
from .python_modules.util import create_dir_if_not_exists, density_scatter, scale_array, distance_x_label
from .python_modules.orbital_similarity import orbital_distance
from .python_modules.structural_similarity import structural_distance
from .algorithm_testing import algo


funColumnMap = {
    # map function : column of row for that function
    orbital_distance : 5
    , structural_distance: 4
}

def sort_by_distance(distance_fun: Callable, descending=False, **kwargs):
    """
    Return the sorted list of all pairs of rows, sorted by the given distance function.

    Defaults to Ascending, for descending pass descending=True.
    """

    funColumnMap = {
        # map function : column of row for that function
        orbital_distance : 5
        , structural_distance: 4
    }

    column_of_interest = funColumnMap[distance_fun]

    ## [(molName, Epm7, Eblyp, smiles, rdk_fingerprints, serialized_molecular_orbital), ...]
    all_ = db.get_all()

    pairs = itertools.combinations(all_, 2)

    distances = []
    for x,y in pairs:
        i = x[column_of_interest]
        j = y[column_of_interest]
        distance = distance_fun(i,j,**kwargs)
        distances.append(
            (
                distance
                , x # (molName, Epm7, Eblyp, smiles, fingerprints, serialized_mol_orb)
                , y # (molName, Epm7, Eblyp, smiles, fingerprints, serialized_mol_orb)
            )
        )

    sortedDistances = sorted(distances, key=lambda x: x[0], reverse=descending)
    
    return sortedDistances

def get_most_least_similar(db: DB, k: int, distance_fun: Callable, **kwargs):
    """
    For each pair of rows (molecules), get the k most and least distant rows based on supplied distance_fun

    Returns most_similar, least_similar
    """

    column_of_interest = funColumnMap[distance_fun]

    ## [(molName, Epm7, Eblyp, smiles, rdk_fingerprints, serialized_molecular_orbital), ...]
    all_ = db.get_all()

    pairs = itertools.combinations(all_, 2)
    
    pairs_map = map(
        lambda pair: (distance_fun(
            pair[0][column_of_interest]
            , pair[1][column_of_interest]
        ),) + pair
        , pairs
    )
    mostDistant, leastDistant = algo(
        pairs_map
        , k=k
        , key=lambda x: x[0]
    )
    
    return mostDistant, leastDistant


def least_most_similar_images(mostSimilar, leastSimilar, outDir: str, distance_fun: Callable):
    """
    mostSimilar and leastSimilar are an array containing tuples of (distance: float, x: DB_row, y: DB_row)
    i.e. for distance, row_x, row_y in mostSimilar:...
    """
    create_dir_if_not_exists(outDir)
    today = datetime.today().strftime("%H-%M-%S")
    outputFile = os.path.join(outDir, f"least_similar_{today}.png")
    draw_grid_images(leastSimilar, distance_fun, outputFile, regression)

    outputFile = os.path.join(outDir, f"most_similar_{today}.png")
    draw_grid_images(mostSimilar, distance_fun, outputFile, regression)


#exit()

def deviation_difference_vs_distance(db:DB, distance_fun: Callable[..., float], resultsDir, show=False, **kwargs):
    ###########################
    """
    For each pair i,j in dataset, calculate a D_ij and Y_ij,
    where D_ij = 1-T(i,j)
    T(i,j) is the tanimoto similarity of molecule i and j

    and

    Y_ij = |dE_i - dE_j| , where dE_z = distance of point z from the linear regression line.
    """
    ###########################
    from itertools import chain

    outfolder = os.path.join(resultsDir, f"Y-vs-D-{distance_fun.__name__}")

    create_dir_if_not_exists(outfolder)
    today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    outfile = os.path.join(outfolder, today)

    ### (molName, Epm7, Eblyp, smiles, fingerprints)
    all_ = db.get_all()
    column_of_interest = funColumnMap[distance_fun]
    pairs = itertools.combinations(all_, 2)

    def fun(i,j):
        D_ij: float = distance_fun(i[column_of_interest],j[column_of_interest], **kwargs)
        Epm7_i, Eblyp_i = (i[1], i[2])
        Epm7_j, Eblyp_j = (j[1], j[2])

        dE_i = regression.distance_from_regress(Epm7_i, Eblyp_i)
        dE_j = regression.distance_from_regress(Epm7_j, Eblyp_j)
        Y_ij: float = abs(dE_i - dE_j)

        return [D_ij,Y_ij]

    length = len(all_)
    print(f"length of all_ = {length}")
    total = int((length * (length-1)) / 2) # n(n-1)/2 is always an integer because n(n-1) is always even.

    map_pairs = np.fromiter(chain.from_iterable(fun(i,j) for i,j in pairs), np.float64, total * 2)
    map_pairs.shape = total, 2

    ### Plot D on x axis, and Y on y axis
    #colours = makeDensityColours(Y)
    #ax.scatter(D, Y,)
    # ax = density_scatter(map_pairs[:,0], map_pairs[:,1], cmap="gnuplot", bins=1000, s=10,)
    #ax.set_title("How Y (= |dE_i - dE_j|) varies with D (= 1-T_ij)\nwhere where dE_z = distance of point z from the linear regression line.")

    X,Y = map_pairs.T

    fig = plt.figure()
    ax = fig.add_subplot()
    h = ax.hist2d(X,Y, bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(f"{distance_x_label(distance_fun)}, D")
    ax.set_ylabel("Difference in energy deviation, Y (eV)")

    regress = linregress(map_pairs[:,0],map_pairs[:,1])
    print(regress)
    plt.tight_layout()
    plt.savefig(outfile + ".png")
    save_distribution(X,Y, distance_fun, outfile + ".csv")
    if show:
        plt.show()

def show_2d_histogram_data(filename):

    folder = os.path.dirname(filename)
    base = os.path.basename(folder)  # 
    distance_fun: str = base.split("-")[-1]
    x_label = distance_fun.replace("_", " ").capitalize()
    print(base, distance_fun)
    with open(filename, "r", newline='') as CsvFile:
        reader = csv.reader(CsvFile)
        X = []
        Y = []
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1])) 

    fig = plt.figure()
    ax = fig.add_subplot()
    h = ax.hist2d(X, Y, bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(f"{x_label}, D")
    ax.set_ylabel("Difference in energy deviation, Y (eV)")
    plt.tight_layout()

    plt.show()

def distance_distribution(db:DB, distance_fun: Callable, resultsDir, show=False, **kwargs):
    """
    Show distribution of distances for chosen distance function. 
    Make sure that column of interest lines up correctly with the chosen distance function,
    I.E. orbital_distance then column should be the serialized molecular orbital column.
    """

    # distribution = {}
    # for distance, row1, row2 in sorted_by_distances:
    #     i = row1[column_of_interest]
    #     j = row2[column_of_interest]
    #     rounded = np.round_(distance, decimals=3)
    #     # c = distribution.get(rounded)
    #     # if c == None:
    #     #     distribution[rounded] = 1
    #     # else:
    #     #     distribution[rounded] += 1
    #     distribution[rounded] = distribution.get(rounded, 0) + 1
        
    ### TODO: Using np.histogram

    column_of_interest = funColumnMap[distance_fun]

    all_ = db.get_all()
    pairs = itertools.combinations(all_, 2)
    distances = np.fromiter(
        (
            distance_fun(
                x[column_of_interest], y[column_of_interest], **kwargs
            ) for x,y in pairs
        )
        , dtype=np.float64
    )
    values, bins = np.histogram(distances, bins='auto')
    Y = values
    X = bins[:-1]
    outfile = os.path.join(
        resultsDir
        , f"{distance_fun.__name__}_distribution"
        , datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    )
    save_distribution(X,Y,distance_fun, outfile + ".csv")

    fig = plt.figure()
    ax = fig.add_subplot()
    width = ( max(X) - min(X) )/ ( len(X)*3 )
    ax.bar(X, Y, width=width)
    #ax.set_title("Distribution of Structural Distances")
    x_label = distance_x_label(distance_fun)
    ax.set_xlabel(f"{x_label}, D")
    ax.set_ylabel("Number of instances")
    plt.tight_layout()
    plt.savefig(outfile + ".png")
    if show:
        plt.show()

def save_distribution(X,Y, distance_fun: Callable, outfile):
    create_dir_if_not_exists(os.path.dirname(outfile))
    with open(outfile, "w", newline='') as CsvFile:
        writer = csv.writer(CsvFile)
        for idx, x in enumerate(X):
            y = Y[idx]
            writer.writerow((x,y))

def show_distribution(filename):

    folder = os.path.dirname(filename)
    base = os.path.basename(folder)  
    distance_fun: str = " ".join(base.split("_")[:-1])
    print(base, distance_fun)
    with open(filename, "r", newline='') as CsvFile:
        reader = csv.reader(CsvFile)
        X = []
        Y = []
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    
    width = ( max(X) - min(X) )/ ( len(X)*2 )
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(X, Y, width=width)
    ax.set_xlabel(f"{distance_fun.capitalize()}, D")
    ax.set_ylabel("Number of instances")
    plt.tight_layout()
    plt.show()



def get_nearest_neighbours(mol_name, k=5,) -> List:
    k_mol_pairs = [x for x in sortedSimilarities if x[1][0] == mol_name or x[2][0] == mol_name][:k]
    neighbours = [(x[2], x[0]) if x[1][0] == mol_name else (x[1],x[0]) for x in k_mol_pairs]
    return neighbours

def main3():
    ##################################
    """
    For each molecule, plot dE vs the avg from the 5 closest molecules.

    for pair in calculated pairs:

    """
    ##################################

    ### TODO similarityMap = {
    #   {mol1, mol2} : similarity(mol1,mol2)
    # }
    # keys are sets, so {mol1,mol2} == {mol2,mol1}

    real = []
    averages = []
    avg_neighb_distances = []
    # boolean array, true if neighbours were > limit, else false
    colours = []
    weighted = False
    for row in all_:
        mol_name, pm7, blyp, smiles, fp = row
        dE_real = regression.distance_from_regress(pm7, blyp)

        neighbours = get_nearest_neighbours(mol_name, k=1)
        dE_neighbours = []
        # each n is same as row
        limit = 0.6
        over_limit = False
        d_k = 1-neighbours[-1][1]
        d_1 = 1-neighbours[0][1]
        for n, similarity in neighbours:
            d_j = 1-similarity
            if d_k == d_1 or weighted:
                w = 1
            else:
                w = (d_k-d_j)/(d_k-1)
            if similarity < limit:
                over_limit=True
            
            neighbour_mol_name, pm7, blyp, smiles, neighbour_fp = n
            pred = regression.distance_from_regress(pm7,blyp) * w
            dE_neighbours.append(pred)

        neighb_distances = [1-x[1] for x in neighbours]
        avg_distance = sum(neighb_distances)/len(neighb_distances)
        avg_neighb_distances.append(avg_distance)
        colours.append(over_limit)
        avg_dE_neighbours = sum(dE_neighbours)/len(dE_neighbours)
        averages.append(avg_dE_neighbours)
        real.append(dE_real)

    ax = plt.subplot()
    ax.scatter(real, averages, c=colours, s=10, cmap=plt.cm.get_cmap('brg'))
    ax.plot(real, real)
    #ax.set_title("How Y (= |dE_i - dE_j|) varies with D (= 1-T_ij)\nwhere where dE_z = distance of point z from the linear regression line.")
    ax.set_xlabel("real deviations")
    ax.set_ylabel("average of deviations nearest neighbours")

    regress = linregress(real, averages)
    print(regress)
    plt.show()

    model_errors = [real[idx]-averages[idx] for idx in range(len(real))]
    ax = plt.subplot()
    ax.scatter(avg_neighb_distances, model_errors)
    #ax.plot(avg_neighb_distances, avg_neighb_distances)
    ax.set_xlabel("average distance of k-neighbours")
    ax.set_ylabel("prediction error (real-predicted) /AU")

    regress = linregress(avg_neighb_distances, model_errors)
    print(regress)
    plt.show()

def save_most_least(mostDistant, leastDistant, outDir):
    """
    mostDistant is k length array of rows (distance, row1, row2)
    pickle this information?
    """
    import pickle
    create_dir_if_not_exists(outDir)
    mostDistantOutFile = os.path.join(outDir, "mostDistant.pkl")
    leastDistantOutFile = os.path.join(outDir, "leastDistant.pkl")
    with open(mostDistantOutFile, "wb") as MostDistantFile, open(leastDistantOutFile, "wb") as LeastDistantFile:
        pickle.dump(mostDistant, MostDistantFile)
        pickle.dump(leastDistant, LeastDistantFile)


def avg_distance_of_k_neighbours(k, db:DB, distance_fun: Callable, resultsDir, ax, show=False):
    """
    Find the k closest neighbours for that point,
    then calculate the average distance.
    
    Or just distribution of avg. distances? <- This one

    To be called for each point in the set.

    Time taken will be n(n-1) * t, 
    
    where n is the size of the set, and t is the time for one comparison.

    Assuming t = 1e-5, and n = 11713

    TotalTime = 11713*11712*1e-5s = 1371s = 22.8mins

    """

    column_of_interest = funColumnMap[distance_fun]

    all_ = db.get_all()

    def fun(x):
        """
        Calculate distance of x to every other point and select
        1:k+1 elements of this ordered list. (0th element is comparing itself)
        Use the numpy argsort to return the indices rather than elements themselves.
        """
        iter = np.fromiter(
            (
                distance_fun(
                    x[column_of_interest], y[column_of_interest], **kwargs
                ) for y in all_
            )
            , np.float64
        )
        neighbours_distances: np.ndarray = np.sort(iter)[1:k+1]
        return sum(neighbours_distances) / len(neighbours_distances)
        
    map_ = map(fun, all_)

    distances = np.fromiter(
        map_
        , dtype=np.float64
    )
    hist = np.histogram(distances, bins='auto')
    Y = hist[0]
    X = hist[1][:-1]

    outfolder = os.path.join(resultsDir, f"avg_{distance_fun.__name__}_of_neighbours_")
    today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    outfile = os.path.join(outfolder, today)
    save_distribution(X, Y, distance_fun, outfile + ".csv")

    width = ( max(X) - min(X) )/ ( len(X)*2 )
    # bar = ax.bar(X,Y,width=1, label=f"k={k}")
    # ax.legend()
    # ax.set_xlabel(f"Average {distance_x_label(distance_fun)} of k Neighbours, " + r'$D_{avg}$')
    # ax.set_ylabel("Number of instances")
    # plt.tight_layout()
    # plt.savefig(outfile + ".png")
    if show:
        plt.show()
    return (distances, k)

def testing_metric(db: DB, funname, distance_fun: Callable, resultsDir:str, n_neighbors=5, **distance_fun_kwargs):
    """
    For each point i, in the dataset, calculate the k nearest neighbours to that point based on the 
    given distance metric (e.g. orbital inertia distance). Then for each nearest neighbour, n_k, calculate
    the difference in their deviations, Y_k = distance_from_regress(E_Pm7(i), E_BLYP(i)) - distance_from_regress(E_Pm7(i), E_BLYP(i))
    i.e. dE_i - d_E_{n_k}
    Calculate the average Y_k for each point i, then plot i vs Y_k

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors
    """
    all_ = np.array(db.get_all())
    column_of_interest = funColumnMap[distance_fun]
    def metric(i,j):
        # i is array containing idx of a row, j is same of another row, return the distance between those rows based on distance_fun
        i = all_[int(i[0])]
        j = all_[int(j[0])]
        return distance_fun(
            i[column_of_interest]
            , j[column_of_interest]
            , **distance_fun_kwargs
        )
    neigh = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
    idxs = [idx for idx in range(len(all_))]
    neigh.fit(np.array([idxs, idxs]).T)
    all_distances, all_indices = neigh.kneighbors(np.array([idxs, idxs]).T) # (array(distances), array(indices))

    Y_averages = []
    avg_distances = []
    dE_pred_list = []
    dE_real_list = []
    for idx, distances in enumerate(all_distances):
        indices = all_indices[idx]

        ### Remove the idx and distance due to comparing against itself
        idx_of_self: Tuple = np.where(indices == idx)
        distances = np.delete(distances, idx_of_self[0][0])
        indices = np.delete(indices, idx_of_self[0][0])


        avg_distance = np.mean(distances)
        _, i_pm7, i_blyp, *_ = all_[idx]
        dE_i = regression.distance_from_regress(i_pm7, i_blyp)
        dE_real_list.append(dE_i)
        neighbor_rows = all_[indices]
        
        dE_k_list = [regression.distance_from_regress(row[1], row[2]) for row in neighbor_rows]

        dE_pred = np.mean(dE_k_list, dtype=np.float64) #type:ignore
        dE_pred_list.append(dE_pred)

        avg_Y = np.mean(
            [abs(dE_k - dE_i) for dE_k in dE_k_list]
            , dtype=np.float64
        ) #type:ignore
        Y_averages.append(avg_Y)
        avg_distances.append(avg_distance)

    ### Save arrays for later plotting
    # today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    outDir = resultsDir
    create_dir_if_not_exists(outDir)
    outfile = os.path.join(outDir, funname + ".npy")
    results = np.array((idxs, Y_averages, avg_distances, dE_pred_list, dE_real_list)).T
    np.save(outfile, results)

def plot_testing_metric_results(filestr, x_max: Optional[float]=None):
    res = np.load(filestr).T
    if x_max:
        results = res.T[np.where(res[2] < x_max)].T
    else:
        results = res

    folder = os.path.dirname(filestr)
    base: str = os.path.basename(filestr)  
    distance_fun = " ".join([x.capitalize() for x in base.split("_")])
    regress = linregress(results[2],results[1])
    print(regress)
    fig = plt.figure()
    ax = fig.add_subplot()
    h = ax.hist2d(results[2],results[1], bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax)
    # ax.scatter(results[2], results[1])

    ax.set_xlabel(distance_fun + r", $\overline{D}_{n,k}$")
    ax.set_ylabel(r"$\overline{Y}_{n,k}$ / eV")

    fig = plt.figure()
    ax2 = fig.add_subplot()
    h = ax2.hist2d(results[2],abs(results[3]-results[4]), bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax2)
    # ax2.scatter(results[2], results[3]-results[4])

    ax2.set_xlabel(distance_fun + r", $\overline{D}_{n,k}$")
    ax2.set_ylabel(r"$|\Delta E_{pred} - \Delta E_{real}|$ / eV")

    plt.show()


def get_small_D_large_Y_from_metric_results(results_file: str, distance_fun, how_many_you_want: int, db:DB,  y_min: float = 1.0, **distance_fun_kwargs):
    """
    Returns the idxs of molecules with small D_{n,k} and large Y_{n,k} when k=1

    results_file should point to metric results .npy file for k=1
    """
    results = np.load(results_file)
    filtered = results[np.where(results.T[1] > y_min)]
    idxs = np.argsort(filtered.T[2])
    sorted_D_ascending = filtered[idxs]
    mol_idxs = sorted_D_ascending.T[0]
    all_ = db.get_all()

    column_of_interest = funColumnMap[distance_fun]
    def metric(i,j):
        # i is array containing idx of a row, j is same of another row, return the distance between those rows based on distance_fun
        i = all_[int(i[0])]
        j = all_[int(j[0])]
        return distance_fun(
            i[column_of_interest]
            , j[column_of_interest]
            , **distance_fun_kwargs
        )
    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    all_idxs = [idx for idx in range(len(all_))]
    nn.fit(np.array([all_idxs, all_idxs]).T)
    all_distances, all_indices = nn.kneighbors(np.array([mol_idxs,mol_idxs]).T)
    output = [] # list of (distance: float, row_x, row_y)
    mol_idxs_already_added_cache = set()
    for idx, distances in enumerate(all_distances):
        indices = all_indices[idx]
        ### Remove the idx and distance due to comparing against itself
        idx_of_self: Tuple = np.where(indices == mol_idxs[idx])
        distances = np.delete(distances, idx_of_self[0][0])
        indices = np.delete(indices, idx_of_self[0][0])

        mol_idx = int(mol_idxs[idx])
        neighbor_mol_idx = int(indices[0])

        if mol_idx not in mol_idxs_already_added_cache and neighbor_mol_idx not in mol_idxs_already_added_cache:
            mol_idxs_already_added_cache.add(mol_idx)
            mol_idxs_already_added_cache.add(neighbor_mol_idx)
        else:
            continue
        

        row_i = all_[mol_idx]
        neighbor_row = all_[neighbor_mol_idx]
        distance = sum(distances)

        # dE_i = regression.distance_from_regress(row_i[1], row_i[2])
        # dE_k = regression.distance_from_regress(neighbor_row[1], neighbor_row[2])
        # Y_nk = abs(dE_i - dE_k)

        output.append((distance, row_i, neighbor_row))
        if len(output) == how_many_you_want:
            break

    return output

def dE_vs_descriptor(
    db:DB
    , descriptor_fun: Callable[[Union[SerializedMolecularOrbital, str]], float]
    , funname: str
    , column: int
    , resultsDir:str
):
    "Plot dE vs value of some descriptor for each molecule"
    all_ = db.get_all_cursor()
    dEs = []
    descriptor_values = []
    for row in all_:
        mol_id, pm7, blyp, smiles, fp, mol_orb = row
        print(mol_id)
        dEs.append(
            regression.distance_from_regress(pm7, blyp)
        )
        descriptor_values.append(
            descriptor_fun(
                row[column]
            )
        )
    outDir = resultsDir
    create_dir_if_not_exists(outDir)
    outfile = os.path.join(outDir, funname + ".npy")
    np.save(outfile, np.array((descriptor_values, dEs)).T)

def plot_dE_vs_descriptor(filestr: str):
    results = np.load(filestr).T
    base: str = os.path.basename(filestr)
    descriptor_fun = os.path.splitext(base)[0]
    regress = linregress(results[0],results[1])
    print(regress)
    fig = plt.figure()
    ax = fig.add_subplot()
    h = ax.hist2d(results[0],results[1], bins=100, cmin=1)
    fig.colorbar(h[3], ax=ax)
    # ax.scatter(results[0], results[1])

    ax.set_xlabel(descriptor_fun)
    ax.set_ylabel(r"$\Delta E$ / eV")
    plt.show()

if __name__ == "__main__":
    #main1()
    #main2()
    #main3()

    

    import sys
    ### Pass distance_fun as arg
    distance_fun_str = sys.argv[1]

    ### Map arg to (distance_fun, kwargs) two-tuple
    distance_fun_map: Mapping[str, Tuple[Callable, dict]] = {
        "inertia_distance": (
            orbital_distance, {
                "inertia_coeff":1
                , "IPR_coeff":0
                , "O_coeff": 0
                , "N_coeff": 0
                , "S_coeff": 0
                , "P_coeff": 0
            }
        )
        , "structural_distance": (
            structural_distance, {}
        )
        , "percent_on_O_distance": (
            orbital_distance, {
                "inertia_coeff":0
                , "IPR_coeff":0
                , "O_coeff": 1
                , "N_coeff": 0
                , "S_coeff": 0
                , "P_coeff": 0
            }
        )
        , "percent_on_N_distance": (
            orbital_distance, {
                "inertia_coeff":0
                , "IPR_coeff":0
                , "O_coeff": 0
                , "N_coeff": 1
                , "S_coeff": 0
                , "P_coeff": 0
            }
        )
        , "percent_on_S_distance": (
            orbital_distance, {
                "inertia_coeff":0
                , "IPR_coeff":0
                , "O_coeff": 0
                , "N_coeff": 0
                , "S_coeff": 1
                , "P_coeff": 0
            }
        )
        , "percent_on_P_distance": (
            orbital_distance, {
                "inertia_coeff":0
                , "IPR_coeff":0
                , "O_coeff": 0
                , "N_coeff": 0
                , "S_coeff": 0
                , "P_coeff": 1
            }
        )
        , "radial_distribution_distance": (
            orbital_distance, {
                "inertia_coeff":0
                , "IPR_coeff":0
                , "O_coeff": 0
                , "N_coeff": 0
                , "S_coeff": 0
                , "P_coeff": 0
                , "radial_distribution_coeff": 1
            }
        )
    }

    n_neighbours = int(sys.argv[2])
    distance_fun, kwargs = distance_fun_map[distance_fun_str]

    today = datetime.today()
    print(today)
    db_path = os.path.join("y4_python", "11k_molecule_database_eV.db")
    db = DB(db_path)
    regression = MyRegression(db)
    #print(f"rmse={regression.rmse}")
    resultsDir = os.path.join("results", "11k_molecule_database_eV", f"n_neigh={n_neighbours}", distance_fun_str)

    # plot_testing_metric_results(
    #     r"results\2021-03-30\11k_molecule_database_eV\n_neigh=1\inertia_distance\inertia_distance.npy"
    #     , x_max=0.03
    # )

    # low_D_large_Y = get_small_D_large_Y_from_metric_results(
    #     r"results\2021-03-30\11k_molecule_database_eV\n_neigh=1\inertia_distance\inertia_distance.npy"
    #     , orbital_distance
    #     , 6
    #     , db
    #     , y_min=1.0
    #     , **{
    #             "inertia_coeff":1
    #             , "IPR_coeff":0
    #             , "O_coeff": 0
    #             , "N_coeff": 0
    #             , "S_coeff": 0
    #             , "P_coeff": 0
    #     }
    # )
    # draw_grid_images(low_D_large_Y, orbital_distance, os.path.join("results","LowDLargeY.png"), regression)

    from .python_modules.descriptors import num_of_atoms, num_of_phosphate_bonds, num_of_sulfate_bonds

    # smiles_column = 3
    # mol_orb_column = 5
    # d = r"results\2021-03-29\11k_molecule_database_eV"
    # for f in os.listdir(d):
    #     plot_testing_metric_results(os.path.join(d,f))

    # for funname, fun, col in (
    #     ( "number of halide atoms", lambda x: num_of_atoms(x, ["F","Cl","Br","I"]), smiles_column )
    #     , ( "number of phosphate bonds", num_of_phosphate_bonds, smiles_column )
    #     , ( "number of sulfate bonds", num_of_sulfate_bonds, smiles_column )
    #     , ( "percent HOMO on N", lambda x: x["percent_on_N"], mol_orb_column )
    #     , ( "percent HOMO on O", lambda x: x["percent_on_O"], mol_orb_column )
    #     , ( "percent HOMO on P", lambda x: x["percent_on_P"], mol_orb_column )
    #     , ( "percent HOMO on S", lambda x: x["percent_on_S"], mol_orb_column )
    # ):
    #     dE_vs_descriptor(
    #         db
    #         , fun #type:ignore  MAKE SURE that fun takes arg of SerializedMolecularOrbital | str and returns float
    #         , funname
    #         , col
    #         , resultsDir
    #     )


    testing_metric(db, distance_fun_str, distance_fun, resultsDir, n_neighbors=n_neighbours, **kwargs)
    # for distance_fun, kwargs in [(orbital_distance, {}), (structural_distance, {})]:
    # for distance_fun, kwargs in [(structural_distance, {})]:
        
        #mostDistant, leastDistant = get_most_least_similar(db, 6, distance_fun, descending=False,)
        #outDir=os.path.join(resultsDir, f"{distance_fun.__name__}_images")
        # save_most_least(mostDistant, leastDistant, outDir)
        # least_most_similar_images(
        #     mostSimilar=leastDistant, leastSimilar=mostDistant, outDir=os.path.join(resultsDir, f"{distance_fun.__name__}_images"), distance_fun=distance_fun
        # )
        # distance_distribution(db, distance_fun, resultsDir, show=True)
        # deviation_difference_vs_distance(db=db, distance_fun=distance_fun, resultsDir=resultsDir, **kwargs)


        # fig = plt.figure()
        # ax = fig.add_subplot()
        # results = []
        # for k in range(5,25,5):
        #     results.append(
        #         avg_distance_of_k_neighbours(k=k, db=db, distance_fun=distance_fun, resultsDir=resultsDir, ax=ax, show=False) # returns (distances, k)
        #     )
        # # ax.hist([x[0] for x in results], bins='auto', label=[f"k={x[1]}" for x in results])
        # # fig.legend()
        # # plt.show()
        # for distances, k in results:
        #     sns.distplot(distances, bins='auto', label=f"k={k}")
        # plt.xlabel(f"Average {distance_x_label(distance_fun)} of k Neighbours, " + r'$D_{avg}$')
        # plt.ylabel("Number of instances")
        # plt.legend()
        # plt.show()
        