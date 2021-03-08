from rdkit import DataStructs
from .util import fingerprint_from_smiles

def similarity_between_two_molecules(s1: str, s2: str, fingerprint_type: int):
    """
    Helper function for quickly printing similarity between two molecules from SMILES

    TODO: Generalise input for different fingerprints? Different Similarity metrics?
    """

    fp1 = fingerprint_from_smiles(s1, fingerprint_type=fingerprint_type)
    fp2 = fingerprint_from_smiles(s2, fingerprint_type=fingerprint_type)

    return DataStructs.FingerprintSimilarity(fp1, fp2 ,metric=DataStructs.TanimotoSimilarity)