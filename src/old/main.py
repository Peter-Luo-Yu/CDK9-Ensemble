import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

from sklearn.decomposition import PCA

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    print(data.columns)
    print(data.head(10))
    data['Active'] = (data['Active'] == 'p').astype(int)
    data['acvalue'] = pd.to_numeric(data['acvalue'], errors='coerce')  # Handle potential non-numeric values
    data = data[data['Std_Smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    return data

def generate_ecfp(smiles, size=1024):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

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
        similarities = [tanimoto_similarity(mol_fp, frag_fp) for frag_fp in fragment_library]
        fragment_similarities.append(similarities)
    
    fragment_matches = [sum(sim > 0.8 for sim in similarities) for similarities in fragment_similarities]

    features = np.hstack((ecfp_features, np.array(fragment_similarities), np.array(fragment_matches).reshape(-1, 1)))
    
    if not (data['acvalue'] == 0).all():
        features = np.hstack((features, data['acvalue'].values.reshape(-1, 1)))
    
    return features
""""
def select_features(X, y):
    selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X, y)
    return selector.transform(X), selector.get_support() """

def tune_hyperparameters(model, X, y):
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, XGBClassifier):
        param_dist = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }
        return RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1],
        }
    elif isinstance(model, MLPClassifier):
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['tanh', 'relu'],
            'alpha': uniform(0.0001, 0.1),
            'learning_rate': ['constant', 'adaptive'],
        }
        return RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
    else:  # LogisticRegression
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],  
            'solver':['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],  # SAGA solver works with both L1 and L2
            'max_iter': [1000, 5000, 10000, 20000],
            'tol': [1e-3, 1e-4, 1e-5, 1e-6]  # Added tolerance parameter
        }
        
    
    return GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def cross_validate_model(model, X, y, cv=5):
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    return auc_scores.mean(), precision_scores.mean()

def create_ensemble(models):
    return VotingClassifier(estimators=list(models.items()), voting='soft')

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('cdk8-smiles.csv')

    # Load fragment library
    fragment_library = load_fragment_library('fragment-smiles.csv')

    # Feature engineering
    features = feature_engineering(data, fragment_library)
    
    # Feature selection
    # X_selected, feature_mask = select_features(features, data['Active'])
    X_selected  = features

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, data['Active'], test_size=0.2, random_state=42)
    
    # Define models
    models = {
        # 'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Multi-layer Perceptron': MLPClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    # Hyperparameter tuning and cross-validation
    results = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        search = tune_hyperparameters(model, X_train, y_train)
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
    
        print(f"Best parameters for {name}: {search.best_params_}")
        print(f"Best score for {name}: {search.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = search.best_estimator_.predict(X_test)
        auc = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
        precision = precision_score(y_test, y_pred)
        results[name] = {'AUC': auc, 'Precision': precision}
    
    # Ensemble model
    ensemble_model = create_ensemble(models)
    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_model.predict_proba(X_test)[:, 1])
    ensemble_precision = precision_score(y_test, y_pred_ensemble)
    results['Ensemble'] = {'AUC': ensemble_auc, 'Precision': ensemble_precision}
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: AUC = {metrics['AUC']:.3f}, Precision = {metrics['Precision']:.3f}")
    
    # Make predictions on all data
    predictions = pd.DataFrame()
    for name, model in models.items():
        all_predictions = model.predict_proba(X_selected)[:, 1]
        predictions[f'{name}_Probability'] = all_predictions
    
    # Add ensemble predictions
    ensemble_predictions = ensemble_model.predict_proba(X_selected)[:, 1]
    predictions['Ensemble_Probability'] = ensemble_predictions
    
    # Add predictions to the original data
    data = pd.concat([data, predictions], axis=1)
    
    # Calculate average probability across all models (including ensemble)
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
    likely_inhibitors.to_csv('likely_cdk8_inhibitors_enhanced.csv', index=False)
    print("\nFull results saved to 'likely_cdk8_inhibitors_enhanced.csv'")
    
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