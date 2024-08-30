from rdkit import Chem
import pandas as pd

def standardize_smiles(smiles):
    """
    Standardize SMILES string to Kekulé representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    else:
        return None  # Return None for invalid SMILES

# Load your datasets
cdk9_data = pd.read_csv('acd990.csv')

# Standardize SMILES for both datasets
cdk9_data['Standardized_SMILES'] = cdk9_data['Smiles'].apply(standardize_smiles)

# Remove any rows where standardization failed (returned None)
cdk9_data = cdk9_data.dropna(subset=['Standardized_SMILES'])

# Print a few examples to verify
print("\nCDK9 Examples:")
print(cdk9_data[['Smiles', 'Standardized_SMILES']].head())

# Save the standardized datasets
cdk9_data.to_csv('acd_standardized.csv', index=False)

print("\nStandardization complete. Check the new CSV files.")