import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
import os

def load_fragment_library(file_path):
    return pd.read_csv(file_path)

def load_trained_models():
    models = {}
    ensemble_model = joblib.load('models/stacking_ensemble_model.pkl')
    models['Stacking Ensemble'] = ensemble_model
    print(f"Loaded model type: {type(ensemble_model)}")
    if hasattr(ensemble_model, 'estimators_'):
        print("Base estimators:")
        for i, est in enumerate(ensemble_model.estimators_):
            print(f"  {i+1}. {type(est).__name__}")
    if hasattr(ensemble_model, 'final_estimator_'):
        print(f"Final estimator: {type(ensemble_model.final_estimator_).__name__}")
    return models

def get_feature_importances(model):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    else:
        return None

def analyze_stacking_classifier(model, fragment_library):
    combined_importances = np.zeros(539)  # Assuming 539 features based on the output
    
    for estimator in model.estimators_:
        importances = get_feature_importances(estimator)
        if importances is not None:
            combined_importances += importances
    
    combined_importances /= len(model.estimators_)
    
    n_fragments = len(fragment_library)
    fragment_importances = combined_importances[-n_fragments:]
    
    top_indices = np.argsort(fragment_importances)[::-1][:10]
    top_fragments = [fragment_library['Fragment'].iloc[i] for i in top_indices]
    top_importances = fragment_importances[top_indices]
    
    return pd.DataFrame({'Fragment': top_fragments, 'Importance': top_importances})

def print_important_fragments(important_fragments):
    print("\nTop 10 Important Fragments:")
    if important_fragments.empty:
        print("No important fragments identified.")
    else:
        for index, row in important_fragments.iterrows():
            print(f"{index + 1}. Fragment: {row['Fragment']}, Importance: {row['Importance']:.4f}")

def plot_important_fragments(important_fragments):
    if important_fragments.empty:
        print("No important fragments to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Fragment', data=important_fragments)
    plt.title('Top 10 Important Fragment Features')
    plt.tight_layout()
    plt.savefig('top_10_fragments_plot.png')
    plt.close()

def save_fragments_to_csv(important_fragments, output_file='top_10_fragments.csv'):
    important_fragments.to_csv(output_file, index=False)
    print(f"Top 10 fragments saved to {output_file}")

def generate_fragment_images(important_fragments, output_dir='fragment_images', model_name='model'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for index, row in important_fragments.iterrows():
        smiles = row['Fragment']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            img_path = os.path.join(output_dir, f"fragment_{index+1}.png")
            img.save(img_path)
            
            axes[index].imshow(img)
            axes[index].axis('off')
            axes[index].set_title(f"Fragment {index+1}", fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(output_dir, f"{model_name}_top_10_fragments.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to free up memory
    print(f"Fragment images and combined figure saved in {output_dir}")

def main():
    fragment_library = load_fragment_library('data/cdk9-fragments.csv')
    print(f"Loaded fragment library. Number of fragments: {len(fragment_library)}")
    
    models = load_trained_models()
    
    if not models:
        print("No models were loaded. Please check the file paths and model files.")
        return
    
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        if isinstance(model, StackingClassifier):
            important_fragments = analyze_stacking_classifier(model, fragment_library)
        else:
            important_fragments = identify_important_fragments(model, fragment_library)
        
        print_important_fragments(important_fragments)
        plot_important_fragments(important_fragments)
        save_fragments_to_csv(important_fragments)
        generate_fragment_images(important_fragments, model_name=name)

if __name__ == "__main__":
    main()