import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import joblib
import csv

def main():
    file_path = "big fat file"

    data = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
    smiles = []

    for index, row in data.iterrows():
        row_string = ' '.join(map(str, row.values))
        if ("Smile" in row_string):
            # print("start: " + data.iloc[index+1])
            smiles.append(data.iloc[index+1])
    
    smiles_df = pd.DataFrame(smiles)
    smiles_df.to_csv('test.csv', index = False)


