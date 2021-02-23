import os
import csv
from typing import List

import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from rdkit import Chem

from PIL.Image import Image

from python_modules.draw_molecule import PILFromSmiles, SMILEStoFiles, concat_images
from python_modules.database import DB

SIG_FIGS = 4
db=DB()

### Read the PM7 and BLYP35 Energy values into memory
### Also read in the molecule names.
PM7File = os.path.join( "sampleInputs", "PM7_energies.txt")
BLYPFile = os.path.join( "sampleInputs","BLYP35_energies.txt")

with open(BLYPFile, 'r', newline='') as F:
    reader = csv.reader(F)
    blyp_data = np.asarray(list(reader)).T

labels = set(blyp_data[0])
blyp_energies = blyp_data[1].astype(np.float)

# as of currently (17/10/2020) the PM7 dataset contains some extras. So we'll filter them out
with open(PM7File, 'r', newline='') as F:
    reader = csv.reader(F)
    filtered = [x for x in reader if x[0] in labels]
    pm7_data = np.asarray(list(filtered)).T


pm7_energies = pm7_data[1].astype(np.float)
pm7_energies = db.get_pm7_energies()

blyp_energies = db.get_blyp_energies()
labels = db.get_mol_names()

### Read the smiles representations of each molecule into a dictionary.
### This gives us O(1) lookup time, and ~1000 entries shouldn't be too memory demanding.
SMILES_file = "python/geoms.smi"
SMILES_dict = {}
with open(SMILES_file, 'r') as F:
    reader = csv.reader(F, dialect='excel-tab')
    for row in reader:
        SMILES_dict[row[1]] = row[0]


def mOnPick(event):
    idx = event.ind[0]
    # # find index of this x value in our x values (CHECK currently blyp)
    mol_name = pm7_data[0][idx]
    diff = x[idx] - y[idx]
    info = f"""{mol_name}, {x[idx]}, {y[idx]}
BLYP35-PM7={np.format_float_scientific(diff, precision=SIG_FIGS)}
SMILES: {SMILES_dict[mol_name]}
    """
    print(info)
    ax.set_title(title_base + "\n" + info)

    ### draw image
    pil = PILFromSmiles(SMILES_dict[mol_name])
    ax3.imshow(pil)

    plt.tight_layout()
    fig.canvas.draw()


fig, (ax, ax2, ax3) = plt.subplots(1,3)
#fig, ax = plt.subplots()
x = pm7_energies
y = blyp_energies
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
### regErrors = [[regression_error_value1, mol_name1],...]
regErrors = (list(map(
    lambda idx: (
        (slope*x[idx] + intercept) - y[idx]
        , pm7_data[0][idx]
    )
    , range(len(y))
)))
### Convert regErrors to absolute values
absRegErrors = np.asarray(list(map(
    lambda x: (
        abs(x[0])
        , x[1]
        )
    , regErrors
)))
points = ax.scatter(x, y, c=absRegErrors.T[0].astype(np.float), picker=True)
yEqualsX, = ax.plot(x, x, color="green")
regressionLine, = ax.plot(x, slope*x+intercept, color="orange")
ax.legend(
    (regressionLine,), ('regression line')
    )

ax.set_xlabel(r'$E_{PM7}$' +" (AU)")
ax.set_ylabel(r'$E_{BLYP}$' +" (AU)")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
title_base = "BLYP35 vs PM7. Click point to print info."
#ax.set_title(title_base)

fig.canvas.mpl_connect('pick_event', mOnPick)

### Add textbox for regression values

textstr = f"""
r = {np.format_float_scientific(r_value, precision=SIG_FIGS)}
p = {np.format_float_scientific(p_value, precision=SIG_FIGS)}
std_err = {np.format_float_scientific(std_err, precision=SIG_FIGS)}
slope = {np.format_float_scientific(slope, precision=SIG_FIGS)}
intercept = {np.format_float_scientific(intercept, precision=SIG_FIGS)}
"""
print(textstr)
props = {
    "boxstyle": "round"
    , "alpha": 0.5
    , "color": "white"
}
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax2.axis('off')


ax3.axis('off')
plt.show()

### TODO: Output most negative deviation molecules
sortedRegErrors = sorted(regErrors, key=lambda x: x[0])
mostNegativeDeviation = sortedRegErrors[:9]

images: List[Image] = SMILEStoFiles(
    [SMILES_dict[x[1]] for x in mostNegativeDeviation]
    , labels=[
        f"{x[1]}\ndE = {np.round_(x[0],decimals=4)}" for x in mostNegativeDeviation
        ]
    )
outFile = os.path.join( "output_mols", "most_negative_deviations.png")
grid_img = concat_images(images)
grid_img.save(outFile)

### TODO: Output most positive deviation molecules
mostPositiveDeviation = sortedRegErrors[-9:]

images: List[Image] = SMILEStoFiles(
    [SMILES_dict[x[1]] for x in mostPositiveDeviation]
    , labels=[
        f"{x[1]}\ndE = {np.round_(x[0],decimals=4)}" for x in mostPositiveDeviation]
    ,
    )
outFile = os.path.join( "output_mols", "most_positive_deviations.png")
grid_img = concat_images(images)
grid_img.save(outFile)