from scipy.stats import linregress

from .database import DB

class MyRegression:
    def __init__(self, db: DB):
        self.db = db
        pm7_energies = db.get_pm7_energies()
        blyp_energies = db.get_blyp_energies()

        x = pm7_energies
        y = blyp_energies

        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = linregress(x,y)

    
    def distance_from_regress(self, x, y):
        """
        y' is y_value from linear regression
        y is real y_value

        return y' - y

        To improve given y value to be on regression line, then
        do: distance_from_regress(x,y) + y 
        """
        y_regress = self.slope*x + self.intercept
        dist = y_regress - y
        return dist