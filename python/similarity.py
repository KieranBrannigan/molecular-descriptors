import csv
import os
from os.path import join
import itertools
from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import _MolsToGridImage
from rdkit.DataStructs.cDataStructs import CreateFromBitString

import numpy as np
from scipy.stats import linregress

from matplotlib import pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from draw_molecule import SMILEStoFiles, concat_images
from database import DB
from regression import distance_from_regress
db = DB()

geomsFile = "geoms.smi"

# with open(geomsFile, 'r') as F:
#     reader = csv.reader(F, dialect='excel-tab')
#     ### row[0] = SMILES, row[1] = molecule_name
    
#     ### morgan fingerprint - # doesn't work
#     #fingerprints = [(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row[0], sanitize=False), 3), row[1], row[0]) for row in reader]
    
#     ### RDK fingerprint - a Daylight-like fingerprint based on hashing molecular subgraphs
#     fingerprints = [(Chem.RDKFingerprint(Chem.MolFromSmiles(row[0], sanitize=False), 3), row[1], row[0]) for row in reader]

## (molName, Epm7, Eblyp, smiles, fingerprints:str)
all_ = db.get_all()
### [(molName, Epm7, Eblyp, smiles, fingerprints:BitVect)]
fingerprints = [row[:-1] + (CreateFromBitString(row[-1]),) for row in all_]

pairs = itertools.combinations(fingerprints, 2)

similarities = []
for x,y in pairs:
    similarity = DataStructs.FingerprintSimilarity(x[-1],y[-1],metric=DataStructs.TanimotoSimilarity)
    similarities.append(
        (
            similarity
            , x # (molName, Epm7, Eblyp, smiles, fingerprints)
            , y # (molName, Epm7, Eblyp, smiles, fingerprints)
        )
    )

sortedSimilarities = sorted(similarities, key=lambda x: x[0], reverse=True)
def main1():
    leastSimilar = np.asarray(sortedSimilarities[-6:])
    mostSimilar = np.asarray(sortedSimilarities[:6])

    def function(array, outputFile):
        images = []
        for similarity,x,y in array:
            x_mol_name, x_pm7, x_blyp, x_smiles, x_fp = x
            y_mol_name, y_pm7, y_blyp, y_smiles, y_fp = y
            mol_x = Chem.MolFromSmiles(x_smiles)
            mol_y = Chem.MolFromSmiles(y_smiles)

            dE_x = distance_from_regress(x_pm7, x_blyp)
            dE_y = distance_from_regress(y_pm7, y_blyp)

            x_description = x_mol_name + "\n" + f"dE = {np.round_(dE_x, decimals=4)}"
            y_description = y_mol_name + "\n" + f"dE = {np.round_(dE_y, decimals=4)}"

            img = _MolsToGridImage([mol_x, mol_y],molsPerRow=2,subImgSize=(400,400))
            #img=Chem.Draw.MolsToGridImage([mol_x, mol_y],molsPerRow=2,subImgSize=(400,400))  
            # fname = join("..","output_mols","least_similar",x[0] + "__" + y[0] + ".png")
            # img.save(fname)
            # img = Image.open(fname)
            draw = ImageDraw.Draw(img)

            mFont = ImageFont.truetype("arial", 32)
            myText = f"similarity = {np.round_(similarity, decimals=3)}"
            draw.text((300, 0),myText,(0,0,0), font=mFont)
            
            mediumFont = ImageFont.truetype("arial", 24)
            draw.text(
                (100, 340)
                , x_description
                , (82,82,82)
                , font=mediumFont)
            draw.text(
                (500, 340)
                , y_description
                , (82,82,82)
                , font=mediumFont)

            images.append(img)
            #img.save(fname)

        grid_img = concat_images(images, num_cols=2)
        grid_img.save(outputFile)

    outputFile = os.path.join("..", "output_mols", "least_similar.png")
    function(leastSimilar, outputFile)

    outputFile = os.path.join("..", "output_mols", "most_similar.png")
    function(mostSimilar, outputFile)


#exit()

def main2():
    ###########################
    """
    For each pair i,j in dataset, calculate a D_ij and Y_ij,
    where D_ij = 1-T(i,j)
    T(i,j) is the tanimoto similarity of molecule i and j

    and

    Y_ij = |dE_i - dE_j| , where dE_z = distance of point z from the linear regression line.
    """
    ###########################
    Y = []
    D = []

    ### (molName, Epm7, Eblyp, smiles, fingerprints)
    all_ = db.get_all()

    pairs = itertools.combinations(all_, 2)

    for i,j in pairs:
        fp_i = CreateFromBitString(i[4])
        fp_j = CreateFromBitString(j[4])
        T_ij = DataStructs.FingerprintSimilarity(fp_i,fp_j,metric=DataStructs.TanimotoSimilarity)
        D_ij = 1-T_ij
        D.append(D_ij)

        Epm7_i, Eblyp_i = (i[1], i[2])
        Epm7_j, Eblyp_j = (j[1], j[2])

        dE_i = distance_from_regress(Epm7_i, Eblyp_i)
        dE_j = distance_from_regress(Epm7_j, Eblyp_j)
        Y_ij = abs(dE_i - dE_j)
        Y.append(Y_ij)

    print(f"D.__len__() = {D.__len__()}")
    ### Plot D on x axis, and Y on y axis
    ax = plt.subplot()
    ax.scatter(D, Y,)
    #ax.set_title("How Y (= |dE_i - dE_j|) varies with D (= 1-T_ij)\nwhere where dE_z = distance of point z from the linear regression line.")
    ax.set_xlabel("Structural Distance, D")
    ax.set_ylabel("Difference in energy deviation, Y (AU)")

    regress = linregress(D,Y)
    print(regress)
    plt.show()

    distribution = {}
    for sim, x, y in similarities:
        rounded = np.round_(1-sim, decimals=2)
        c = distribution.get(rounded)
        if c == None:
            distribution[rounded] = 1
        else:
            distribution[rounded] += 1
        
    ax = plt.subplot()
    ax.bar(distribution.keys(), distribution.values(), width=0.002)
    #ax.set_title("Distribution of Structural Distances")
    ax.set_xlabel("Structural Distance, D")
    ax.set_ylabel("Number of instances / Probability density")
    plt.tight_layout()
    plt.show()




def get_nearest_neighbours(mol_name, k=5,) -> List:
    k_mol_pairs = [x for x in sortedSimilarities if x[1][0] == mol_name or x[2][0] == mol_name][:k]
    neighbours = [(x[2], x[0]) if x[1][0] == mol_name else (x[1],x[0]) for x in k_mol_pairs]
    return neighbours

def main_3():
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
        dE_real = distance_from_regress(pm7, blyp)

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
            pred = distance_from_regress(pm7,blyp) * w
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


if __name__ == "__main__":
    main_3()