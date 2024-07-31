import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    print(data.columns)
    print(data.head(10))
    data['Active'] = (data['Active'] == 'p').astype(int)
    data['acvalue'] = data['acvalue'].astype(float)
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

def select_features(X, y):
    selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X, y)
    return selector.transform(X), selector.get_support()

def create_logistic_regression_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(random_state=42))
    ])

def tune_hyperparameters(model, X, y):
    if isinstance(model, Pipeline):
        if 'logistic' in model.named_steps:
            param_grid = {
                'sgd__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,10,100],
                'sgd__penalty': ['l2', 'l1', 'elasticnet'],
                'sgd__max_iter': [100, 500, 1000],
                'sgd__tol': [1e-4, 1e-3, 1e-2],
                'sgd__loss': ['log', 'modified_huber']
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
            'gamma': ['scale', 'auto', 0.1, 1],
        }
    elif isinstance(model, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
    else:
        raise ValueError("Unsupported model type")
    
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
    X_selected, feature_mask = select_features(features, data['Active'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, data['Active'], test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': create_logistic_regression_pipeline(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Multi-layer Perceptron': MLPClassifier(random_state=42), 
     #   'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)  # Using GPU for XGBoost
    }

    # Hyperparameter tuning and cross-validation
    results = {}
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        tuned_model = tune_hyperparameters(model, X_train, y_train)
        tuned_model.fit(X_train, y_train)
        best_model = tuned_model.best_estimator_
        models[name] = best_model

        # Save the best model parameters
        joblib.dump(best_model, f'{name}_best_model.pkl')

        print(f"Best parameters for {name}: {tuned_model.best_params_}")
        print(f"Best score for {name}: {tuned_model.best_score_:.3f}")

        print(f"Cross-validating {name}...")
        y_pred = tuned_model.best_estimator_.predict(X_test)
        auc = roc_auc_score(y_test, tuned_model.best_estimator_.predict_proba(X_test)[:, 1])
        precision = precision_score(y_test, y_pred)
        results[name] = {'AUC': auc, 'Precision': precision}
        
    
    # Create and fit ensemble model
    ensemble_model = create_ensemble(models)
    print("Fitting Ensemble model...")
    ensemble_model.fit(X_selected, data['Active'])
    print("Cross-validating Ensemble model...")
    ensemble_auc, ensemble_precision = cross_validate_model(ensemble_model, X_selected, data['Active'])
    results['Ensemble'] = {'AUC': ensemble_auc, 'Precision': ensemble_precision}
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: AUC = {metrics['AUC']:.3f}, Precision = {metrics['Precision']:.3f}")
    
    # Save the ensemble model
    joblib.dump(ensemble_model, 'ensemble_model.pkl')

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