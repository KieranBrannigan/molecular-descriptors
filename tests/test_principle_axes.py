import unittest
import json
from os.path import dirname, join

import matplotlib.pyplot as plt

from y4_python.python_modules.orbital_calculations import MolecularOrbital

class TestPrincipleAxes(unittest.TestCase):

    def test_plot_unit_square(self):

        # load test "orbital" unit square

        testDir = dirname(__file__)

        fig = plt.figure()

        testfile = join(testDir, "unit_square_orbital.json")
        mo = MolecularOrbital.fromJsonFile(testfile, MolecularOrbital.HOMO, molecule_name="unit square")
        mo.plot(mo.molecule_name, 111, fig)
        print(mo.principle_axes)

        plt.show()
