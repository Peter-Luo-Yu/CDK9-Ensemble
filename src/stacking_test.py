import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, header = None, encoding = 'latin-1', on_bad_lines = 'skip')
    print(data.head())  
    print(f"Number of rows read: {len(data)}")
    return data

# Function to generate ECFP
def generate_ecfp(smiles, size=1024):
    try:
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        
        # Check if molecule conversion was successful
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Generate Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=size)
        return fingerprint

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

def tanimoto_similarity(fp1, fp2):
    if fp1 is None or fp2 is None:
        print("One or both fingerprints are None.")
        return None
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Function to load fragment library
def load_fragment_library(file_path):
    fragments = pd.read_csv(file_path)
    fragment_fps = []
    for smiles in fragments['Fragment']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fragment_fps.append(fp)
    return fragment_fps

# Function to perform feature engineering
def feature_engineering(data, fragment_library):
    ecfp_features = []
    fragment_similarities = []
    fragment_matches = []
    valid_smiles = []

    for smiles in data:
        ecfp = generate_ecfp(smiles)
        if ecfp is not None:
            ecfp_features.append(ecfp)
            
            similarities = [tanimoto_similarity(ecfp, frag_fp) for frag_fp in fragment_library]
            fragment_similarities.append(similarities)
            fragment_matches.append(sum(sim > 0.8 for sim in similarities))
            
            valid_smiles.append(smiles)
        else:
            print(f"Skipping invalid SMILES: {smiles}")
    
    ecfp_features = np.array(ecfp_features)
    fragment_similarities = np.array(fragment_similarities)
    fragment_matches = np.array(fragment_matches).reshape(-1, 1)

    features = np.hstack((ecfp_features, fragment_similarities, fragment_matches))
    
    print("Feature dimensions:", features.shape)
    return features, valid_smiles

# Main function to load the model and make predictions
def main(smiles_file, fragment_library_file):
    # Load new SMILES data
    new_data = load_and_preprocess_data(smiles_file)
    print(f"Processed data shape: {new_data.shape}")
    
    # Load fragment library
    fragment_library = load_fragment_library(fragment_library_file)
    print(f"Loaded {len(fragment_library)} fragments")

    # Feature engineering
    print("Starting feature engineering...")
    smiles_data = new_data.iloc[:, 0]
    
    new_features, valid_smiles = feature_engineering(smiles_data, fragment_library)
    print(f"Feature engineering complete. Features shape: {new_features.shape}")

    # Load feature selector and transform features
    print("Loading feature selector...")
    selector = joblib.load('models/feature_selector.pkl')
    new_features_selected = selector.transform(new_features)
    print(f"Selected features shape: {new_features_selected.shape}")

    # Load ensemble model
    print("Loading ensemble model...")
    ensemble_model = joblib.load('models/stacking_ensemble_model.pkl')

    # Predict using the ensemble model
    print("Making predictions...")
    predictions = ensemble_model.predict(new_features_selected)
    prediction_probabilities = ensemble_model.predict_proba(new_features_selected)[:, 1]
    print(prediction_probabilities)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of prediction probabilities: {len(prediction_probabilities)}")

    # Create a new DataFrame with predictions
    results_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Prediction': predictions,
        'Prediction_Probability': prediction_probabilities
    })

    # Sort the DataFrame by prediction probability in descending order
    sorted_df = results_df.sort_values(by='Prediction_Probability', ascending=False)

    # Save the sorted DataFrame to predictions.csv
    predictions_file = 'soft_predictions.csv'
    sorted_df.to_csv(predictions_file, index=False, float_format='%.10f')
    print("Sorted predictions saved to", predictions_file)
    print(f"Final dataframe shape: {sorted_df.shape}")

"""
    print(len(results_df))
    for i in range(len(results_df)):
        smile = sorted_df['SMILES'].iloc[i]
        print(str(i) + " " + smile)
        mol = Chem.MolFromSmiles(smile)
        filename = str(i) + ".png"
        Draw.MolToFile(mol, filename)
"""

if __name__ == "__main__":
    main('test/test_all_2.csv', 'data/cdk9-fragments.csv')