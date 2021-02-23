import os
from typing import List, Iterable

from rdkit.Chem import Draw, MolFromSmiles, AllChem, Mol, SanitizeMol, SanitizeFlags

from PIL import Image, ImageFont, ImageDraw

from .util import create_dir_if_not_exists, sanitize_without_hypervalencies

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

def MolsToFiles(molList: Iterable[Mol], size=(400,400), labels: Iterable[str]=None) -> List[Image.Image]:
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