import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import joblib

# Must be the same both times
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    data['Active'] = (data['Active'] == 'p').astype(int)
    data['acvalue'] = data['acvalue'].astype(float)
    data = data[data['Std_Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    return data

# Function to generate ECFP
def generate_ecfp(smiles, size=1024):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)

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
    ecfp_features = np.array([generate_ecfp(smiles) for smiles in data['Std_Smiles']])
    fragment_similarities = []
    for smiles in data['Std_Smiles']:
        mol_fp = generate_ecfp(smiles)
        similarities = [DataStructs.TanimotoSimilarity(mol_fp, frag_fp) for frag_fp in fragment_library]
        fragment_similarities.append(similarities)

    fragment_matches = [sum(sim > 0.8 for sim in similarities) for similarities in fragment_similarities]

    features = np.hstack((ecfp_features, np.array(fragment_similarities), np.array(fragment_matches).reshape(-1, 1)))

    if not (data['acvalue'] == 0).all():
        features = np.hstack((features, data['acvalue'].values.reshape(-1, 1)))
    

    print("Feature dimensions:", features.shape)
    return features

# Main function to load the model and make predictions
def main(smiles_file, fragment_library_file):
    # Load new SMILES data
    new_data = load_and_preprocess_data(smiles_file)
    # Load fragment library
    fragment_library = load_fragment_library(fragment_library_file)

    # Feature engineering
    new_features = feature_engineering(new_data, fragment_library)

    # Load feature selector and transform features
    selector = joblib.load('models/feature_selector.pkl')
    new_features_selected = selector.transform(new_features)

    # Load ensemble model
    ensemble_model = joblib.load('models/ensemble_model.pkl')

    # Predict using the ensemble model
    predictions = ensemble_model.predict(new_features_selected)
    prediction_probabilities = ensemble_model.predict_proba(new_features_selected)[:, 1]

    # Add predictions to the dataframe
    new_data['Prediction'] = predictions
    new_data['Prediction_Probability'] = prediction_probabilities

    # Sort the DataFrame by prediction probability in descending order
    sorted_df = new_data.sort_values(by='Prediction_Probability', ascending=False)

    # Save the sorted DataFrame to predictions.csv
    predictions_file = 'predictions.csv'
    sorted_df.to_csv(predictions_file, index=False)
    print("Sorted predictions saved to", predictions_file)


if __name__ == "__main__":
    main('data/cdk8-smiles.csv', 'data/fragment-smiles.csv')