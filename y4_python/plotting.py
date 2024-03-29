import os
import csv
from typing import List
from y4_python.python_modules.orbital_calculations import MolecularOrbital

import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from rdkit import Chem

from PIL.Image import Image

from .python_modules.draw_molecule import PILFromSmiles, SMILEStoFiles, concat_images, resize_image
from .python_modules.database import DB
from .python_modules.util import density_scatter
from .python_modules.regression import MyRegression

def main(db:DB):
    SIG_FIGS = 4
    mol_orbitals_dir = r"sampleInputs\11k_orbitals"

    my_regression = MyRegression(db)
    ### Read the PM7 and BLYP35 Energy values into memory
    ### Also read in the molecule names and smiles.
    pm7_energies = db.get_pm7_energies()

    blyp_energies = db.get_blyp_energies()
    labels = db.get_mol_ids()
    smiles = db.get_smiles()

    orbital_fig = plt.figure()
    def mOnPick(event):
        idx = event.ind[0]
        # # find index of this x value in our x values (CHECK currently blyp)
        mol_name = labels[idx]

        mo = MolecularOrbital.fromJsonFile(
            os.path.join(mol_orbitals_dir, mol_name + ".json")
            , MolecularOrbital.HOMO
        ).plot(mol_name=mol_name, axis_number=111, fig=orbital_fig)
        orbital_fig.canvas.draw()
        
        diff = x[idx] - y[idx]
        info = f"""{mol_name}, {x[idx]}, {y[idx]}
    BLYP35-PM7={np.format_float_scientific(diff, precision=SIG_FIGS)}
    SMILES: {smiles[idx]}
        """
        print(info)
        ax.set_title(title_base + "\n" + info)

        ### draw image
        pil = PILFromSmiles(smiles[idx])
        ax3.imshow(pil)

        plt.tight_layout()
        fig.canvas.draw()


    fig, (ax, ax2, ax3) = plt.subplots(1,3)
    # fig, ax = plt.subplots()
    x = pm7_energies
    y = blyp_energies
    x = np.array(x)
    y = np.array(y) 
    ### regErrors = [[regression_error_value1, mol_name1],...]
    regErrors = (list(map(
        lambda idx: (
            my_regression.distance_from_regress(x[idx],y[idx])
            , labels[idx]
            , smiles[idx]
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
    #ax = density_scatter(x, y, ax, fig=fig, picker=True)
    #yEqualsX, = ax.plot(x, x, color="green")
    regressionLine, = ax.plot(x, my_regression.slope*x+my_regression.intercept, color="orange")
    # ax.legend(
    #     (regressionLine,), ('regression line')
    #     )

    ax.set_xlabel(r'$E_{PM7}$' +" (eV)")
    ax.set_ylabel(r'$E_{BLYP}$' +" (eV)")
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    title_base = "BLYP35 vs PM7. Click point to print info."
    #ax.set_title(title_base)

    fig.canvas.mpl_connect('pick_event', mOnPick)

    ### Add textbox for regression values

    textstr = f"""
    r = {np.format_float_scientific(my_regression.r_value, precision=SIG_FIGS)}
    p = {np.format_float_scientific(my_regression.p_value, precision=SIG_FIGS)}
    std_err = {np.format_float_scientific(my_regression.std_err, precision=SIG_FIGS)}
    slope = {np.format_float_scientific(my_regression.slope, precision=SIG_FIGS)}
    intercept = {np.format_float_scientific(my_regression.intercept, precision=SIG_FIGS)}
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
        [x[2] for x in mostNegativeDeviation]
        , labels=[
            f"{x[1]}\nΔE = {np.round_(x[0],decimals=4)} eV" for x in mostNegativeDeviation
            ]
        )
    outFile = os.path.join( "output_mols", "most_negative_deviations.png")
    grid_img = concat_images(images)
    grid_img = resize_image(grid_img, 800)
    grid_img.save(outFile)

    ### TODO: Output most positive deviation molecules
    mostPositiveDeviation = sortedRegErrors[-9:]

    images: List[Image] = SMILEStoFiles(
        [x[2] for x in mostPositiveDeviation]
        , labels=[
            f"{x[1]}\nΔE = {np.round_(x[0],decimals=4)} eV" for x in mostPositiveDeviation]
        ,
        )
    outFile = os.path.join( "output_mols", "most_positive_deviations.png")
    grid_img = concat_images(images)
    grid_img = resize_image(grid_img, 800)
    grid_img.save(outFile)

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1]
    db = DB(db_path)
    main(db)