import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
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
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Multi-layer Perceptron': MLPClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

def tune_hyperparameters(model, X, y):
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    elif isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
    elif isinstance(model, XGBClassifier):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    elif isinstance(model, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    else:
        raise ValueError("Unsupported model type")
    
    return GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def create_stacking_ensemble(models):
    meta_model = LogisticRegression(random_state=42)
    return StackingClassifier(
        estimators=list(models.items()),
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba'
    )

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_precision = precision_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_precision = precision_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return {
        'Train AUC': train_auc, 'Train Precision': train_precision, 'Train Accuracy': train_accuracy,
        'Test AUC': test_auc, 'Test Precision': test_precision, 'Test Accuracy': test_accuracy
    }

def plot_auc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Load and preprocess data
    data = load_and_preprocess_data('data/cdk9-smiles.csv')
    fragment_library = load_fragment_library('data/cdk9-fragments.csv')
    features = feature_engineering(data, fragment_library)
    
    # Feature selection
    X_selected, selector = select_features(features, data['Active'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, data['Active'], test_size=0.2, random_state=42)
    
    # Create models
    models = create_models()
    
    # Hyperparameter tuning and training
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        tuned_model = tune_hyperparameters(model, X_train, y_train)
        tuned_model.fit(X_train, y_train)
        best_model = tuned_model.best_estimator_
        models[name] = best_model

        print(f"Best parameters for {name}: {tuned_model.best_params_}")
        print(f"Best score for {name}: {tuned_model.best_score_:.3f}")
    
    # Create and train stacking ensemble
    stacking_ensemble = create_stacking_ensemble(models)
    print("Training Stacking Ensemble...")
    stacking_ensemble.fit(X_train, y_train)
    
    # Add stacking ensemble to models dictionary
    models['Stacking Ensemble'] = stacking_ensemble
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Print results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Train: AUC = {metrics['Train AUC']:.3f}, Precision = {metrics['Train Precision']:.3f}, Accuracy = {metrics['Train Accuracy']:.3f}")
        print(f"  Test:  AUC = {metrics['Test AUC']:.3f}, Precision = {metrics['Test Precision']:.3f}, Accuracy = {metrics['Test Accuracy']:.3f}")
    
    # Plot AUC curves for all models
    plot_auc_curves(models, X_test, y_test)

if __name__ == "__main__":
    main()