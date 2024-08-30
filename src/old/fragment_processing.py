from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def is_valid_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        return True
    except:
        return False

# Read your data (adjust the file name and format as needed)
df = pd.read_csv('data/cdk9-fragments-formatted.csv')

# Assuming your SMILES are in a column named 'SMILES'
df['is_valid'] = df['Fragment'].apply(is_valid_mol)

# Separate valid and invalid compounds
valid_df = df[df['is_valid']]
invalid_df = df[~df['is_valid']]

# Save the results
valid_df.to_csv('valid_compounds.csv', index=False)
invalid_df.to_csv('invalid_compounds.csv', index=False)

print(f"Total compounds: {len(df)}")
print(f"Valid compounds: {len(valid_df)}")
print(f"Invalid compounds: {len(invalid_df)}")