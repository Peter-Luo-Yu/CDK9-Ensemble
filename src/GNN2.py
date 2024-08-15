import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class GNNWithFragments(nn.Module):
    def __init__(self, num_features, num_fragments):
        super().__init__()
        self.conv1 = nn.Linear(num_features, 64)
        self.conv2 = nn.Linear(64, 64)
        self.conv3 = nn.Linear(64, 64)
        self.fc1 = nn.Linear(64 + num_fragments, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, fragment_features):
        row, col = edge_index
        for conv in [self.conv1, self.conv2, self.conv3]:
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            x = F.relu(conv(out))
        
        x = x.mean(dim=0)
        x = torch.cat([x, fragment_features], dim=0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return torch.sigmoid(x)

def mol_to_features_and_edge_index(mol):
    num_atoms = mol.GetNumAtoms()
    features = np.zeros((num_atoms, 4))
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        features[idx, 0] = atom.GetAtomicNum()
        features[idx, 1] = int(atom.GetIsAromatic())
        features[idx, 2] = atom.GetDegree()
        features[idx, 3] = atom.GetTotalNumHs()

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.LongTensor(edges).t().contiguous()

    return torch.FloatTensor(features), edge_index

def load_fragments(file_path):
    df = pd.read_csv(file_path)
    fragments = df['Fragment'].tolist()
    return [Chem.MolFromSmiles(fragment) for fragment in fragments]

def get_fragment_features(mol, fragments):
    return torch.FloatTensor([float(mol.HasSubstructMatch(frag)) for frag in fragments])

def predict_activity(model, smiles, fragments, device):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    
    features, edge_index = mol_to_features_and_edge_index(mol)
    fragment_features = get_fragment_features(mol, fragments)
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    fragment_features = fragment_features.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(features, edge_index, fragment_features)
    
    return prediction.item()

def load_data(file_path):
    df = pd.read_csv(file_path)
    smiles = df['Std_Smiles'].tolist()

    df['Active'] = (df['Active'] == 'p').astype(int)
    activities = df['Active'].tolist()
    return smiles, activities

def train_model(model, train_data, fragments, device, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Shuffle the training data
        random.shuffle(train_data)

        for smiles, activity in train_data:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            features, edge_index = mol_to_features_and_edge_index(mol)
            fragment_features = get_fragment_features(mol, fragments)
            
            features = features.to(device)
            edge_index = edge_index.to(device)
            fragment_features = fragment_features.to(device)
            activity = torch.FloatTensor([activity]).to(device)
            
            optimizer.zero_grad()
            prediction = model(features, edge_index, fragment_features)
            loss = criterion(prediction, activity)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_data):.4f}")

def main():
    device = torch_directml.device()
    
    # Load fragments
    fragments_file = 'data/fragment-smiles.csv'
    fragments = load_fragments(fragments_file)
    
    # Load training data
    data_file = 'data/cdk8-smiles.csv'
    smiles, activities = load_data(data_file)
    
    # Split data into train and test sets
    train_smiles, test_smiles, train_activities, test_activities = train_test_split(
        smiles, activities, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = GNNWithFragments(num_features=4, num_fragments=len(fragments)).to(device)
    train_data = list(zip(train_smiles, train_activities))
    train_model(model, train_data, fragments, device)
    
    # Save the trained model
    torch.save(model.state_dict(), 'gnn_fragment_bioactivity_model.pth')
    
    # Make predictions on test set
    test_predictions = []
    for smiles in test_smiles:
        prediction = predict_activity(model, smiles, fragments, device)
        test_predictions.append(prediction)
    
    # Calculate and print test set accuracy
    test_accuracy = sum(1 for pred, true in zip(test_predictions, test_activities) if round(pred) == true) / len(test_activities)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Make predictions on new data
    new_smiles_file = 'test/test_BBs.csv'
    new_smiles = pd.read_csv(new_smiles_file, header = None)[0].tolist()
    
    results = []
    for smiles in new_smiles:
        activity_prediction = predict_activity(model, smiles, fragments, device)
        results.append({
            'SMILES': smiles,
            'Predicted_Activity': activity_prediction
        })

    results_df = pd.DataFrame(results)
    sorted_df = results_df.sort_values(by='Predicted_Activity', ascending=False)
    predictions_file = 'fragment_enhanced_predictions.csv'
    sorted_df.to_csv(predictions_file, index=False)
    
    print(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    main()