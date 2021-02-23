### This tests that converting fingerprint to BitString, writing it to database and reading it back
### before converting back to BitVect (fingerprint) will not produce different Tanimoto similarities.

from itertools import combinations
import sqlite3
import unittest
from random import sample
import numpy as np

from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs.cDataStructs import BitVectToText, CreateFromBitString

from y4_python.python_modules.database import DB
from y4_python.python_modules.util import sanitize_without_hypervalencies

class TestDatabase(unittest.TestCase):

    def setUp(self) -> None:
        test_db = sqlite3.connect(":memory:")
        self.c = test_db.cursor()
        self.c.execute(
            "CREATE TABLE data (fingerprints text)"
        )

        self.smiles: np.ndarray  = DB().get_smiles()
        self.smiles_sample = sample(list(self.smiles), 50)


        self.pairs = combinations(self.smiles_sample, 2)

    def test_fingerprint_bit_string_readwrite(self):
        """
        Test that the similarity of two molecules isn't changed when we 
        write and then read the fingerprint (as a BitString) from the database.
        """
        total = 0
        for samples in self.pairs:
            mol_x = Chem.MolFromSmiles(samples[0], sanitize=False)
            mol_y = Chem.MolFromSmiles(samples[1], sanitize=False)

            sanitize_without_hypervalencies(mol_x)
            sanitize_without_hypervalencies(mol_y)

            fp_x = Chem.RDKFingerprint(mol_x)
            fp_y = Chem.RDKFingerprint(mol_y)

            similarity_before = FingerprintSimilarity(fp_x, fp_y)

            bitString_x = BitVectToText(fp_x)
            bitString_y = BitVectToText(fp_y)

            self.c.execute("INSERT INTO data VALUES (?)", (bitString_x,))
            self.c.execute("INSERT INTO data VALUES (?)", (bitString_y,))

            read_x = self.c.execute(f"SELECT * FROM data WHERE fingerprints='{bitString_x}'").fetchone()[0]
            read_y = self.c.execute(f"SELECT * FROM data WHERE fingerprints='{bitString_y}'").fetchone()[0]

            new_fp_x = CreateFromBitString(read_x)
            new_fp_y = CreateFromBitString(read_y)

            similarity_after = FingerprintSimilarity(new_fp_x, new_fp_y)
            
            if abs(similarity_after - similarity_before) > 0.01:
                print(f"similarity before = {similarity_before}, similarity after = {similarity_after}")
                total += 1
        
        self.assertTrue(total == 0
            , f"Some Similarities were not equal, total={total}" 
        )
        
if __name__ == "__main__":
    unittest.main()