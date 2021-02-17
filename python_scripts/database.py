import os
import sqlite3
import csv
from typing import List

import numpy as np
from rdkit.DataStructs.cDataStructs import BitVectToText, CreateFromBitString

from util import fingerprint_from_smiles

# for idx, row in enumerate(blyp_data):
#     if row[0] == pm7_data[idx][0]:
#         blyp_data[idx] = tuple(row[0] + pm7_data[idx][1] + row[1])
#     else:
#         print(f"line {idx} , molNames not the same, blyp name = {row[0]} pm7 name = {pm7_data[idx][0]}")

class DB:

    BLYP = 'E_blyp'
    PM7 = 'E_pm7'

    def __init__(self):
        self.conn = sqlite3.connect("D:\Projects\Y4Project\python\energies_database.db")
        self.cur = self.conn.cursor()

    def table_exists(self):
        r = self.cur.execute("SELECT name FROM sqlite_master WHERE type=`table` AND name=`dataset`")
        if len(r.fetchall()) == 1:
            return True
        else:
            return False

    def create_table(self):
        self.cur.execute("CREATE TABLE IF NOT EXISTS dataset (mol_name text, E_pm7 real, E_blyp real, smiles text, fingerprints text)")

    def add_dataset(self, dataset):
        for name, pm7, blyp, smiles, fingerprint_bitvect in dataset:
            self.cur.execute(
                "INSERT INTO dataset VALUES (?,?,?,?,?)", (name, float(pm7), float(blyp), smiles, fingerprint_bitvect)
            )
        self.commit()
        

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    ######################
    ### Reading Operations
    ######################

    def get_all(self):
        r = self.cur.execute(
            "SELECT * FROM dataset ORDER BY `rowid`"
        )
        return r.fetchall()

    def get_row_from_mol_name(self, mol_name):
        r = self.cur.execute(
            f"SELECT * FROM dataset WHERE mol_name='{mol_name}'"
        )
        fetch = r.fetchall()
        if fetch.__len__() > 1:
            print(f"WARNING: fetching row for  mol_name {mol_name} gave more than one row result")
        return fetch[0]

    def get_dE_from_mol_name(self, mol_name):
        row = self.get_row_from_mol_name(mol_name)
        pm7, blyp = row[1:3]
        dE = blyp - pm7
        return dE

    def base_get_energies(self, energy: str) -> List[float]:
        "energy: str = self.PM7 or self.BLYP"
        col = "E_pm7" if energy == self.PM7 else "E_blyp"
        r = self.cur.execute(
            f"SELECT {col} FROM dataset ORDER BY `rowid`"
        )
        ### Reshape [(val,), (val2,)...] into [val, val2,...]
        return np.asarray(r.fetchall())[:,0]

    def get_blyp_energies(self) -> List[float]:
        "Return BLYP energies sorted by rowid"
        return self.base_get_energies(self.BLYP)

    def get_pm7_energies(self) -> List[float]:
        "Return PM7 energies sorted by rowid"
        return self.base_get_energies(self.PM7)

    def get_pm7_energies_with_smiles(self):
        r = self.cur.execute(
            f"SELECT `{self.PM7}`, `smiles` FROM dataset ORDER BY `rowid`"
        )
        return r.fetchall()

    def get_mol_names(self) -> List[str]:
        "Return mol_names ordered by rowid"
        r = self.cur.execute(
            "SELECT mol_name FROM dataset ORDER BY `rowid`"
        )
        return r.fetchall()

    def get_smiles(self) -> List[str]:
        "Return smiles ordered by rowid"
        r = self.cur.execute(
            "SELECT `smiles` FROM dataset ORDER BY `rowid`"
        )
        return np.asarray(r.fetchall())[:,0]

    def get_smiles_for_mol(self, mol_name):
        r = self.cur.execute(
            f"SELECT `smiles` WHERE `mol_name`={mol_name}"
        )
        return r.fetchone()

    def get_fingerprints(self):
        r = self.cur.execute(
            "SELECT `fingerprints` FROM dataset ORDER BY `rowid`"
        )
        return list(map(
            lambda x: CreateFromBitString(x[0])
            , r.fetchall()
        ))


def main():

    inputDir = os.path.join("..","sampleInputs")
    BLYP_file = "D:\Projects\Y4Project\sampleInputs\BLYP35_energies.txt"
    PM7_file = "D:\Projects\Y4Project\sampleInputs\PM7_energies.txt"

    with open(BLYP_file, 'r', newline='') as F:
        reader = csv.reader(F)
        ### data is [[molName, E_blyp], ...]
        blyp_data = np.asarray([tuple(row) for row in reader])

    mol_names = blyp_data.T[0]
    blypEnergies = blyp_data.T[1]
    with open(PM7_file, 'r', newline='') as F:
        pm7_reader = list(csv.reader(F))
        ### Each pm7_row in reader is [molName, E_blyp]
        # we filter out rows that didn't exist in the blyp data.
        pm7_data = [tuple(row) for row in pm7_reader if row[0] in mol_names]

    pm7Energies = np.asarray(pm7_data).T[1]

    db = DB()

    db.create_table()

    ### Read the smiles representations of each molecule into a dictionary.
    ### This gives us O(1) lookup time, and ~1000 entries shouldn't be too memory demanding.
    SMILES_file = "D:\Projects\Y4Project\python\geoms.smi"
    SMILES_dict = {}
    with open(SMILES_file, 'r') as F:
        reader = csv.reader(F, dialect='excel-tab')
        for row in reader:
            SMILES_dict[row[1]] = row[0]

    ### This is memory intensive, however it makes sure there is no error.
    SMILES_list = [SMILES_dict[mol_name] for mol_name in mol_names]
    fingerprint__bitvect_list = np.asarray([
        BitVectToText(fingerprint_from_smiles(smiles)) for smiles in SMILES_list
    ])
    dataset = []
    for idx in range(len(mol_names)):
        dataset.append((mol_names[idx], pm7Energies[idx], blypEnergies[idx], SMILES_list[idx], fingerprint__bitvect_list[idx]))
    ### Add dataset
    #dataset = np.asarray([mol_names, pm7Energies, blypEnergies, SMILES_list, fingerprint__bitvect_list]).T
    db.add_dataset(dataset)

    db.close()

if __name__ == "__main__":
    main()