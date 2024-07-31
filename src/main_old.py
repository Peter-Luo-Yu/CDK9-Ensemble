import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score

from sklearn.decomposition import PCA

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, sep=',')
    print(data.columns)
    print(data.head(10))
    # Convert 'Active' column to binary (0 for 'n', 1 for 'p')
    data['Active'] = (data['Active'] == 'p').astype(int)
    
    # Convert acvalue to float if it's not already
    data['acvalue'] = data['acvalue'].astype(float)
    
    # Remove rows with invalid SMILES
    data = data[data['Std_Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    return data

def generate_ecfp(smiles, size=1024):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def load_fragment_library(file_path):
    # Load fragments from CSV file
    fragments = pd.read_csv(file_path)
    # Generate fingerprints for each fragment
    fragment_fps = []
    for smiles in fragments['Fragment']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fragment_fps.append(fp)

    return fragment_fps

def feature_engineering(data, fragment_library):
    ecfp_features = np.array([generate_ecfp(smiles) for smiles in data['Std_Smiles']]) #ECFP from the SMILES from the CDK8 data

    fragment_similarities = [] # is a 2d array, with each element being a list of tanimoto scores
    for smiles in data['Std_Smiles']:       
        mol_fp = generate_ecfp(smiles)                                                         # Again, but for only 1 SMILE
        similarities = [tanimoto_similarity(mol_fp, frag_fp) for frag_fp in fragment_library]  # Caculates all the tanimoto scores for that 1 SMILE and ALL the Fragments
        fragment_similarities.append(similarities)  
    
    fragment_matches = [sum(sim > 0.8 for sim in similarities) for similarities in fragment_similarities] # number of fragments that have greater than 0.8 score for each SMILE

    features = np.hstack((ecfp_features, np.array(fragment_similarities), np.array(fragment_matches).reshape(-1, 1)))
    
    # Add 'acvalue' as a feature if it's not zero for all compounds
    if not (data['acvalue'] == 0).all():
        features = np.hstack((features, data['acvalue'].values.reshape(-1, 1)))
    
    return features

def create_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    precision = precision_score(y_test, y_pred)
    return auc, precision

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('cdk8-smiles.csv')

    # Load fragment library
    fragment_library = load_fragment_library('fragment-smiles.csv')

    # Feature engineering
    features = feature_engineering(data, fragment_library)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, data['Active'], test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Multi-layer Perceptron': MLPClassifier(),
        'XGBoost': XGBClassifier()
    }
    
    
    # Train models and make predictions
    results = {}
    predictions = pd.DataFrame()
    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        results[name] = {'AUC': auc, 'Precision': precision}
        
        # Make predictions on all data
        all_predictions = model.predict_proba(features)[:, 1]
        predictions[f'{name}_Probability'] = all_predictions
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: AUC = {metrics['AUC']:.3f}, Precision = {metrics['Precision']:.3f}")
    
    # Add predictions to the original data
    data = pd.concat([data, predictions], axis=1)
    
    # Calculate average probability across all models
    probability_columns = [col for col in data.columns if col.endswith('_Probability')]
    data['Average_Probability'] = data[probability_columns].mean(axis=1)
    
    # Sort by average probability
    likely_inhibitors = data.sort_values('Average_Probability', ascending=False)
    
    # Print top 10 likely inhibitors
    print("\nTop 10 likely CDK8 inhibitors (based on average probability):")
    top_10 = likely_inhibitors[['Std_Smiles', 'Average_Probability'] + probability_columns].head(10)
    pd.set_option('display.max_columns', None)
    print(top_10)
    
    # Save results to CSV
    likely_inhibitors.to_csv('likely_cdk8_inhibitors_all_models.csv', index=False)
    print("\nFull results saved to 'likely_cdk8_inhibitors_all_models.csv'")
    
    # Analyze model agreement
    threshold = 0.95
    agreement_count = (predictions > threshold).sum(axis=1)
    high_agreement = data[agreement_count == len(models)]
    print(f"\nNumber of compounds with high agreement (all models predict > {threshold}): {len(high_agreement)}")
    
    if len(high_agreement) > 0:
        print("\nTop 5 compounds with high model agreement:")
        print(high_agreement.sort_values('Average_Probability', ascending=False)[['Std_Smiles', 'Average_Probability']].head())

if __name__ == "__main__":
    main()