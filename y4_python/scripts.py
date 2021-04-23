from matplotlib import pyplot as plt
from y4_python.python_modules.orbital_similarity import OrbitalDistanceKwargs, orbital_distance
import numpy as np
from timeit import timeit
from itertools import combinations

from sklearn import metrics

from y4_python.similarity import plot_testing_metric_results, plot_metric_test_threshold
from y4_python.python_modules.structural_similarity import structural_distance
from y4_python.python_modules.database import DB
from y4_python.python_modules.regression import MyRegression
from y4_python.python_modules.chemical_distance_metric import MetricParams, chemical_distance
from y4_python.learning import main_chemical_distance, show_results
from y4_python.learning import main_euclidean_distance

def plot_testing_results(relative_file_path, x_max=None):
    db = DB("y4_python/11k_molecule_database_eV.db")
    reg = MyRegression(db)

    plot_testing_metric_results(relative_file_path, reg, x_max=x_max)

def D_RDF_from_mol_ids(mol_id1,mol_id2):
    from y4_python.python_modules.orbital_similarity import radial_distribution_difference
    db = DB("y4_python/11k_molecule_database_eV.db")
    row1 = db.get_row_from_mol_id(mol_id1)
    row2 = db.get_row_from_mol_id(mol_id2)
    i_mol_id, i_pm7, i_blyp, i_smiles, i_fp, i_homo, i_lumo = row1
    j_mol_id, j_pm7, j_blyp, j_smiles, j_fp, j_homo, j_lumo = row2
    return radial_distribution_difference(i_homo, j_homo)

def Delta_Ei_from_mol_id(mol_id):
    db = DB("y4_python/11k_molecule_database_eV.db")
    regression = MyRegression(db)
    row1 = db.get_row_from_mol_id(mol_id)
    i_mol_id, i_pm7, i_blyp, i_smiles, i_fp, i_homo, i_lumo = row1
    return regression.distance_from_regress(i_pm7, i_blyp)

def dE_from_row_idx(row_idx):
    db = DB("y4_python/11k_molecule_database_eV.db")
    regression = MyRegression(db)
    all_ = db.get_all()
    row1 = all_[row_idx]
    i_mol_id, i_pm7, i_blyp, i_smiles, i_fp, i_homo, i_lumo = row1
    return regression.distance_from_regress(i_pm7, i_blyp)

def re_arrange_learning_results(results_file):
    res = np.load(results_file) # [[y_real, y_pred], [...], ...]
    db = DB("y4_python/11k_molecule_database_eV.db")
    regression = MyRegression(db)
    m_r, c_r = regression.slope, regression.intercept
    all_ = db.get_all()

    def get_Eblyp(row_idx, dE_pred):
        _, i_pm7, *_ = all_[row_idx]
        return (m_r*i_pm7 + c_r) - dE_pred
        
    res = res[np.argsort(res[:,0])]

    E_blyp_pred = np.fromiter(
        ( get_Eblyp(res[idx][0], res[idx][2]) for idx in range(len(res)) )
        , dtype=np.float64
    )

    array_all = np.array(all_)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(array_all[:,1], array_all[:,2])
    ax2 = fig.add_subplot()
    ax2.scatter(array_all[:,1], E_blyp_pred)

    plt.show()  
    
    
def euc(i,j):
    return sum(
        (i[idx]-j[idx])**2 for idx in range(len(i))
    ) ** 0.5

def time_chemical_distance_metric():
    ""
    db = DB("y4_python/11k_molecule_database_eV.db")
    all_ = db.get_all_cursor()
    fingerprint_list = db.get_fingerprints()
    homo_list = db.get_homo_molecular_orbitals()
    lumo_list = db.get_lumo_molecular_orbitals()
    idx = 0
    rows = []

    i = np.array([0])
    j = np.array([1])


    def run_euc():
        return euc(i,j)

    def run(it):
        for a,b in it:
            i = np.array([a])
            j = np.array([b])
            return chemical_distance(
                i
                , j
                , homo_coeff=1.0
                , lumo_coeff=0.0
                , fingerprint_list=fingerprint_list
                , homo_orbital_list=homo_list
                , lumo_orbital_list=lumo_list
                , c_orbital=1.0
                , c_struct=1.0
                , inertia_coefficient=0.0
                , IPR_coefficient=0.0
                , N_coefficient=0.0
                , O_coefficient=0.0
                , S_coefficient=0.0
                , P_coefficient=0.0
                , radial_distribution_coeff=1.0
            )
    number = 100_000
    it_range = 10
    it = combinations(range(it_range), 2)
    print(f"len(it) = {it_range} * ({it_range-1}) / 2 = {it_range*(it_range-1) / 2}")
    print(timeit(lambda: run(it), number=number) / number)

    print(timeit(lambda: run_euc(), number=number) / number)


def time_euclidean_distance_learning():
    "run learning with Euc distance to see the time taken and compare with chemical_distance"
    
    db = DB("y4_python/11k_molecule_database_eV.db")
    reg = MyRegression(db)
    mol_list = db.get_mol_ids()
    pm7_energies = db.get_pm7_energies()
    blyp_energies = db.get_blyp_energies()
    deviation_list = (list(map(reg.distance_from_regress, pm7_energies, blyp_energies)))
    
    main_euclidean_distance(
        k_neighbours=5
        , k_folds=10
        , metric_params={}
        , mol_list=mol_list
        , deviation_list=deviation_list
    )

def time_chemical_distance_learning():
    "run learning with different params to see what takes the longest. Use limited set of data so it doesn't take so long on each run."

    db = DB("y4_python/11k_molecule_database_eV.db")
    reg = MyRegression(db)

    cutoff = 1000

    mol_list = db.get_mol_ids()[:cutoff]
    pm7_energies = db.get_pm7_energies()[:cutoff]
    blyp_energies = db.get_blyp_energies()[:cutoff]
    deviation_list = (list(map(reg.distance_from_regress, pm7_energies, blyp_energies)))
    fingerprint_list = db.get_fingerprints()[:cutoff]
    homo_list = db.get_homo_molecular_orbitals()[:cutoff]
    lumo_list = db.get_lumo_molecular_orbitals()[:cutoff]
    
    params = MetricParams(
        homo_coeff=1.0
        , lumo_coeff=0.0
        , fingerprint_list=fingerprint_list
        , homo_orbital_list=homo_list
        , lumo_orbital_list=lumo_list
        , c_struct=0.0
        , c_orbital=1.0
        , inertia_coefficient=0.0
        , IPR_coefficient=0.0
        , N_coefficient=0.0
        , O_coefficient=0.0
        , S_coefficient=0.0
        , P_coefficient=0.0
        , radial_distribution_coeff=1.0
    )

    results = main_chemical_distance(
        k_neighbours=5
        , k_folds=10
        , metric_params=params
        , mol_list=mol_list
        , deviation_list=deviation_list
        , save=False
    )
    print(results)

def time_structural_distance():
    db = DB("y4_python/11k_molecule_database_eV.db")
    fingerprint_list = db.get_fingerprints()

    def run():
        ctr = 0
        for idx in range(0, len(fingerprint_list), 2):
            try:
                j = fingerprint_list[idx+1]
            except:
                break
            i = fingerprint_list[idx]
            structural_distance(i,j)
            ctr+=1
        print("counter = ", ctr)

    number=1
    t = timeit(lambda: run(), number=number)
    print(t)

def time_RDF_distance():
    db = DB("y4_python/11k_molecule_database_eV.db")

    homo_list = db.get_homo_molecular_orbitals()
    lumo_list = []

    def run():
        ctr = 0
        for idx in range(0, len(homo_list), 2):
            try:
                j = homo_list[idx+1]
            except:
                break
            i = homo_list[idx]
            orbital_distance(
                i, {}
                , j, {}
                ,homo_coeff=1.0
                , lumo_coeff=0.0
                , orbital_distance_kwargs={"radial_distribution_coeff":1.0}
            )
            ctr+=1
        #print("counter = ", ctr)

    number=1000
    t = timeit(lambda: run(), number=number)
    print(t)

def time_RDF_and_structural():
    db = DB("y4_python/11k_molecule_database_eV.db")
    all_ = db.get_all()
    fp_list = db.get_fingerprints()
    homo_list = db.get_homo_molecular_orbitals()
    lumo_list = db.get_lumo_molecular_orbitals()

    def run():
        ctr = 0
        for idx in range(0, len(all_), 2):
            try:
                j = np.array([idx+1])
            except:
                break
            i = np.array([idx])
            chemical_distance(
                i
                , j
                , homo_coeff=1.0
                , lumo_coeff=0.0
                , fingerprint_list=fp_list
                , homo_orbital_list=homo_list
                , lumo_orbital_list=lumo_list
                , **{"c_struct":1.0, "c_orbital":1.0, "radial_distribution_coeff":1.0}
            )
            ctr+=1
        #print("counter = ", ctr)

    number=100
    t = timeit(lambda: run(), number=number)
    print(t)

def time_euc():
    db = DB("y4_python/11k_molecule_database_eV.db")
    all_ = db.get_all()
    fp_list = db.get_fingerprints()
    homo_list = db.get_homo_molecular_orbitals()
    lumo_list = db.get_lumo_molecular_orbitals()

    i = np.array([1000, 1000, 1000])
    j = np.array([2000, 2000, 2000])
    def run():
        ctr = 0
        for idx in range(0, len(all_), 2):
            euc(i, j)
            ctr+=1
        #print("counter = ", ctr)

    number=100
    t = timeit(lambda: run(), number=number)
    print(t)

if __name__ == "__main__":
    #show_results(r"results\2021-04-20\learning_21-47-05.npy")
    # time_chemical_distance_learning()
    # plot_testing_results(r"results\2021-04-20\learning_21-47-05.npy")
    time_euc()