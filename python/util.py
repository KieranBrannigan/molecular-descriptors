import os

from rdkit.Chem import Mol, SanitizeFlags, SanitizeMol, MolFromSmiles, RDKFingerprint

from rdkit.DataStructs.cDataStructs import BitVectToBinaryText, CreateFromBinaryText

def create_dir_if_not_exists(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        return

def sanitize_without_hypervalencies(m: Mol):
    ### Sanitize molecule (without checking for hypervalencies)
    SanitizeMol(
        m
        ,SanitizeFlags.SANITIZE_FINDRADICALS|SanitizeFlags.SANITIZE_KEKULIZE|SanitizeFlags.SANITIZE_SETAROMATICITY|SanitizeFlags.SANITIZE_SETCONJUGATION|SanitizeFlags.SANITIZE_SETHYBRIDIZATION|SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True
        )

def fingerprint_from_smiles(s):
    m = MolFromSmiles(s, sanitize=False)
    sanitize_without_hypervalencies(m)
    return RDKFingerprint(m)
