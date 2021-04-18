import os
from typing import Callable, List, Iterable
from y4_python.python_modules.structural_similarity import structural_distance
import numpy as np

from rdkit import Chem
from rdkit.Chem.Draw import _MolsToGridImage

from rdkit.Chem import Draw, MolFromSmiles, AllChem, Mol, SanitizeMol, SanitizeFlags

from PIL import Image, ImageFont, ImageDraw

from y4_python.python_modules.database import DB
from y4_python.python_modules.regression import MyRegression
from .util import create_dir_if_not_exists, sanitize_without_hypervalencies, distance_x_label

def PILFromSmiles(smiles:str):
    """
    Input: str  smiles representation of a molecule
    Output: PIL image 2d representation of the input molecule
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not calculate molecule from smiles. Make sure smiles is correct?\nsmiles input was: {smiles}")
    ### Not sure if needed
    #tmp=AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(mol)

def add_label_to_image(img: Image.Image, text:str, fontsize:int=24, pos=(0.25, 0.85), color=(82,82,82)):
    size = img.size
    x, y = (pos[0]*size[0], pos[1]*size[1])
    draw = ImageDraw.Draw(img)
    mediumFont = ImageFont.truetype("arial", fontsize)
    draw.text(
        (100, 340)
        , text
        , color
        , font=mediumFont)

def MolsToFiles(molList: Iterable[Mol], size=(400,400), labels: List[str]=None) -> List[Image.Image]:
    out = []
    for idx,m in enumerate(molList):
        if m is None:
            print(f"index {idx} m was none. Was the smiles input correct?")
            continue
        
        sanitize_without_hypervalencies(m)
        img = Draw.MolToImage(m, size, kekulize=False)
        add_label_to_image(img, text=labels[idx])
        out.append(img)        
    return out

def SMILEStoFiles(smilesList: Iterable[str], size=(400,400),  labels: Iterable[str]=None) -> List[Image.Image]:
    molList = [MolFromSmiles(x, sanitize=False) for x in smilesList]
    for m in molList: sanitize_without_hypervalencies(m)
    return MolsToFiles(molList, size=size, labels=labels)

def concat_images(images: List[Image.Image], num_cols=3):
    """
    Assumes all images are same size (w & h)
    """
    each_width, each_height = images[0].width, images[0].height
    dst_width = num_cols*images[0].width
    num_rows = len(images)//num_cols + 1 if len(images) % num_cols > 0 else len(images)//num_cols
    dst_height = num_rows*images[0].height

    dst = Image.new('RGB', (int(dst_width), int(dst_height)))

    start=0
    for row in range(num_rows):
        end = (row+1)*num_cols
        row_images = images[start:end]
        for idx, img in enumerate(row_images):
            dst.paste(
                img, (idx*each_width, row*each_height)
                )
        start=end

    return dst

def resize_image(image: Image.Image, new_width) -> Image.Image:
    "Resize image maintaining aspect ratio."
    wpercent = (new_width/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    new_image = image.resize((new_width,hsize), Image.ANTIALIAS)
    return new_image

def draw_grid_images(array, distance_fun: Callable, outputFile: str, regression: MyRegression, ):
    """
    array is list of distance: float, x: DB_row, y: DB_row
    """
    images = []
    for distance,x,y in array:
        x_mol_name, x_pm7, x_blyp, x_smiles, x_fp, x_serialized_homo, x_serialized_lumo = x
        y_mol_name, y_pm7, y_blyp, y_smiles, y_fp, y_serialized_homo, y_serialized_lumo = y

        struct_distance = structural_distance(x_fp, y_fp)

        mol_x = Chem.MolFromSmiles(x_smiles)
        mol_y = Chem.MolFromSmiles(y_smiles)

        dE_x = regression.distance_from_regress(x_pm7, x_blyp)
        dE_y = regression.distance_from_regress(y_pm7, y_blyp)

        x_description = x_mol_name + "\n" + f"ΔE = {np.round_(dE_x, decimals=4)} eV"
        y_description = y_mol_name + "\n" + f"ΔE = {np.round_(dE_y, decimals=4)} eV"

        subImgSize = (400, 400)
        img = _MolsToGridImage([mol_x, mol_y],molsPerRow=2,subImgSize=subImgSize)
        #img=Chem.Draw.MolsToGridImage([mol_x, mol_y],molsPerRow=2,subImgSize=(400,400))  
        # fname = join("..","output_mols","least_similar",x[0] + "__" + y[0] + ".png")
        # img.save(fname)
        # img = Image.open(fname)
        draw = ImageDraw.Draw(img)

        W, H = subImgSize
        ### Draw thin rectangle around img
        draw.rectangle(
            [0,0,W*2,H]
            , width = 2
            , outline="#000000"
        )

        mFont = ImageFont.truetype("arial", 32)
        myText = f"{distance_x_label(distance_fun)} = {np.round_(distance, decimals=4)} \nStructural Distance = {np.round_(struct_distance, decimals=4)} \nY_ij = {np.round_(abs(dE_x - dE_y), decimals=4)}"
        w,h = draw.textsize(myText, mFont)
        draw.text(
            (W-w/2, 0),myText,(0,0,0), font=mFont
        )
        
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
    resize_image(grid_img, 800)
    grid_img.save(outputFile)