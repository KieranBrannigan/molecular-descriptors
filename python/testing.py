### This tests that converting fingerprint to BitString, writing it to database and reading it back
### before converting back to BitVect (fingerprint) will not produce different Tanimoto similarities.

from itertools import combinations
import sqlite3

from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs.cDataStructs import BitVectToText, CreateFromBitString

from database import DB
from util import sanitize_without_hypervalencies

new_db = sqlite3.connect(":memory:")
c = new_db.cursor()
c.execute(
    "CREATE TABLE data (fingerprints text)"
)

smiles = DB().get_smiles()

pairs = combinations(smiles, 2)

total = 0
for samples in pairs:
    mol_x = Chem.MolFromSmiles(samples[0], sanitize=False)
    mol_y = Chem.MolFromSmiles(samples[1], sanitize=False)

    sanitize_without_hypervalencies(mol_x)
    sanitize_without_hypervalencies(mol_y)

    fp_x = Chem.RDKFingerprint(mol_x)
    fp_y = Chem.RDKFingerprint(mol_y)

    similarity_before = FingerprintSimilarity(fp_x, fp_y)

    bitString_x = BitVectToText(fp_x)
    bitString_y = BitVectToText(fp_y)

    c.execute("INSERT INTO data VALUES (?)", (bitString_x,))
    c.execute("INSERT INTO data VALUES (?)", (bitString_y,))

    read_x = c.execute(f"SELECT * FROM data WHERE fingerprints='{bitString_x}'").fetchone()[0]
    read_y = c.execute(f"SELECT * FROM data WHERE fingerprints='{bitString_y}'").fetchone()[0]

    new_fp_x = CreateFromBitString(read_x)
    new_fp_y = CreateFromBitString(read_y)

    similarity_after = FingerprintSimilarity(new_fp_x, new_fp_y)
    
    if abs(similarity_after - similarity_before) > 0.01:
        print(f"similarity before = {similarity_before}, similarity after = {similarity_after}")
        total += 1
    print(total)
