import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    data['Active'] = (data['Active'] == 'p').astype(int)
    data['acvalue'] = data['acvalue'].astype(float)
    data = data[data['Std_Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
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
    ecfp_features = np.array([generate_ecfp(smiles) for smiles in data['Std_Smiles']])
    fragment_similarities = []
    for smiles in data['Std_Smiles']:
        mol_fp = generate_ecfp(smiles)
        similarities = [DataStructs.TanimotoSimilarity(mol_fp, frag_fp) for frag_fp in fragment_library]
        fragment_similarities.append(similarities)
    
    fragment_matches = [sum(sim > 0.8 for sim in similarities) for similarities in fragment_similarities]

    features = np.hstack((ecfp_features, np.array(fragment_similarities), np.array(fragment_matches).reshape(-1, 1)))
    
    #if not (data['acvalue'] == 0).all():
    #    features = np.hstack((features, data['acvalue'].values.reshape(-1, 1)))
    
    print("ecfp_features shape:", ecfp_features.shape)
    print("fragment_similarities_array shape:", len(fragment_similarities), " ", len(fragment_similarities[0]))
    print("fragment_matches_array shape:", np.array(fragment_matches).reshape(-1, 1).shape)
    print("Combined features shape:", features.shape)
    print("\nCombined features:")
    print(features)
    
    return features

def select_features(X, y):
    selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X, y)
    return selector.transform(X), selector

def create_logistic_regression_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(random_state=42))
    ])

def tune_hyperparameters(model, X, y):
    if isinstance(model, Pipeline):
        if 'logistic' in model.named_steps:
            param_grid = {
                'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'logistic__penalty': ['l2'],
                'logistic__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                'logistic__max_iter': [100,500,1000],
                'logistic__tol': [1e-3, 1e-4, 1e-5, 1e-6]
            }
    elif isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, XGBClassifier):
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    elif isinstance(model, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    else:
        raise ValueError("Unsupported model type")
    
    return GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def create_soft_ensemble(models):
    return VotingClassifier(estimators=list(models.items()), voting='soft')

def cross_validate_model(model, X, y, cv=5):
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    return auc_scores.mean(), precision_scores.mean()


def main():
    # Load and preprocess data
    data = load_and_preprocess_data('data/cdk8-smiles.csv')

    # Load fragment library
    fragment_library = load_fragment_library('data/fragment-smiles.csv')

    # Feature engineering
    features = feature_engineering(data, fragment_library)
    
    
    # Feature selection
    X_selected, selector = select_features(features, data['Active'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, data['Active'], test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': create_logistic_regression_pipeline(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Multi-layer Perceptron': MLPClassifier(random_state=42)
    }
    
    # Hyperparameter tuning and cross-validation
    results = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        tuned_model = tune_hyperparameters(model, X_train, y_train)
        tuned_model.fit(X_train, y_train)
        best_model = tuned_model.best_estimator_
        models[name] = best_model

        print(f"Best parameters for {name}: {tuned_model.best_params_}")
        print(f"Best score for {name}: {tuned_model.best_score_:.3f}")

        print(f"Cross-validating {name}...")
        y_pred = tuned_model.best_estimator_.predict(X_test)
        auc = roc_auc_score(y_test, tuned_model.best_estimator_.predict_proba(X_test)[:, 1])
        precision = precision_score(y_test, y_pred)
        results[name] = {'AUC': auc, 'Precision': precision}
    
    # Create and fit soft ensemble model
    soft_ensemble_model = create_soft_ensemble(models)  
    print("Fitting Ensemble model...")
    soft_ensemble_model.fit(X_selected, data['Active'])
    print("Cross-validating Ensemble model...")
    soft_ensemble_auc, soft_ensemble_precision = cross_validate_model(soft_ensemble_model, X_selected, data['Active'])
    results['Ensemble'] = {'AUC': soft_ensemble_auc, 'Precision': soft_ensemble_precision}
    
    # Save models and feature selector
    for name, model in models.items():
        joblib.dump(model, f'{name}_best_model.pkl')
    joblib.dump(soft_ensemble_model, 'ensemble_model.pkl')
    joblib.dump(selector, 'feature_selector.pkl')
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: AUC = {metrics['AUC']:.3f}, Precision = {metrics['Precision']:.3f}")

if __name__ == "__main__":
    main()