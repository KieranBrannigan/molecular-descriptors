from scipy.stats import linregress

from database import DB

db = DB()

pm7_energies = db.get_pm7_energies()
blyp_energies = db.get_blyp_energies()

x = pm7_energies
y = blyp_energies

slope, intercept, r_value, p_value, std_err = linregress(x,y)

def distance_from_regress(x, y):
    y_regress = slope*x + intercept
    dist = y_regress - y
    return dist