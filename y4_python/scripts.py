from y4_python.similarity import plot_testing_metric_results
from y4_python.python_modules.database import DB
from y4_python.python_modules.regression import MyRegression

def plot_testing_results(relative_file_path, x_max=None):
    db = DB("y4_python/11k_molecule_database_eV.db")
    reg = MyRegression(db)

    plot_testing_metric_results(relative_file_path, reg, x_max=x_max)