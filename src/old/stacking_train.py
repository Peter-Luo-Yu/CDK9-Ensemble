import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    #data['Active'] = (data['Active'] == 'p').astype(int)
    data = data[data['Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    return data

def generate_ecfp(smiles, size=1024):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)

def load_fragment_library(file_path):
    fragments = pd.read_csv(file_path)
    fragment_fps = []
    for smiles in fragments['Fragment']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fragment_fps.append(fp)
    return fragment_fps

def feature_engineering(data, fragment_library):
    ecfp_features = np.array([generate_ecfp(smiles) for smiles in data['Smiles']])
    fragment_similarities = []
    for smiles in data['Smiles']:
        mol_fp = generate_ecfp(smiles)
        similarities = [DataStructs.TanimotoSimilarity(mol_fp, frag_fp) for frag_fp in fragment_library]
        fragment_similarities.append(similarities)
    
    fragment_matches = [sum(sim > 0.8 for sim in similarities) for similarities in fragment_similarities]

    features = np.hstack((ecfp_features, np.array(fragment_similarities), np.array(fragment_matches).reshape(-1, 1)))
    return features

def select_features(X, y):
    selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X, y)
    return selector.transform(X), selector

def create_models():
    return {
        'Random Forest': RandomForestClassifier(random_state=42),
        #'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Multi-layer Perceptron': MLPClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

def create_stacking_ensemble(models):
    meta_model = LogisticRegression(random_state=42)
    return StackingClassifier(
        estimators=list(models.items()),
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba'
    )

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {'AUC': auc, 'Precision': precision, 'Accuracy': accuracy}

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('data/cdk9-smiles.csv')
    fragment_library = load_fragment_library('data/cdk9-fragments.csv')
    features = feature_engineering(data, fragment_library)
    
    # Feature selection
    X_selected, selector = select_features(features, data['Active'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, data['Active'], test_size=0.2, random_state=42)
    
    # Create and train models
    models = create_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    # Create and train stacking ensemble
    stacking_ensemble = create_stacking_ensemble(models)
    print("Training Stacking Ensemble...")
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test)
    
    print("Evaluating Stacking Ensemble...")
    results['Stacking Ensemble'] = evaluate_model(stacking_ensemble, X_test, y_test)
    
    # Print results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: AUC = {metrics['AUC']:.3f}, Precision = {metrics['Precision']:.3f}, Accuracy = {metrics['Accuracy']:.3f}")
    
    # Save models and feature selector
    for name, model in models.items():
        joblib.dump(model, f'{name}_model.pkl')
    joblib.dump(stacking_ensemble, 'stacking_ensemble_model.pkl')
    joblib.dump(selector, 'feature_selector.pkl')

if __name__ == "__main__":
    main()