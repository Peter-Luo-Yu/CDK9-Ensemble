import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from rdkit import Chem
import numpy as np
import pandas as pd

# Recreate the GNN class structure
class GNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Linear(num_features, 64)
        self.conv2 = nn.Linear(64, 64)
        self.conv3 = nn.Linear(64, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index):
        # Simple message passing
        row, col = edge_index
        for conv in [self.conv1, self.conv2, self.conv3]:
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            x = F.relu(conv(out))
        
        # Global mean pooling
        x = x.mean(dim=0)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return torch.sigmoid(x)

def mol_to_features_and_edge_index(mol):
    # Get atom features
    num_atoms = mol.GetNumAtoms()
    features = np.zeros((num_atoms, 4))  # 4 features: atomic_num, is_aromatic, degree, num_h
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        features[idx, 0] = atom.GetAtomicNum()
        features[idx, 1] = int(atom.GetIsAromatic())
        features[idx, 2] = atom.GetDegree()
        features[idx, 3] = atom.GetTotalNumHs()

    # Get bond information (edge_index)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))  # add reverse edge for undirected graph

    edge_index = torch.LongTensor(edges).t().contiguous()

    return torch.FloatTensor(features), edge_index

def predict_activity(model, smiles, device):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    
    features, edge_index = mol_to_features_and_edge_index(mol)
    features, edge_index = features.to(device), edge_index.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(features, edge_index)
    
    return prediction.item()

def load_smiles_from_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Assuming the SMILES are in a column named 'Std_Smiles'
    # If the column name is different, please change it accordingly
    if 'Std_Smiles' in df.columns:
        return df['Std_Smiles'].tolist()
    else:
        print(f"Error: 'Std_Smiles' column not found in {file_path}")
        print(f"Available columns: {df.columns.tolist()}")
        return []

def main():
    # Set up the device
    device = torch_directml.device()
    
    # Load the trained model
    model = GNN(num_features=4)  # Make sure this matches your trained model architecture
    model.load_state_dict(torch.load('gnn_bioactivity_model.pth', map_location=device))
    model = model.to(device)
    
    # Load SMILES from file
    smiles_file = 'data/cdk8-smiles.csv'  # Update this path
    test_smiles = load_smiles_from_file(smiles_file)
    
    if not test_smiles:
        print("No SMILES loaded. Please check the file path and column name.")
        return

    # Prepare a list to store results
    results = []

    for smiles in test_smiles:
        activity_prediction = predict_activity(model, smiles, device)
        
        # Append result to the list
        results.append({
            'SMILES': smiles,
            'Predicted_Activity': activity_prediction
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort the DataFrame by Predicted_Activity in descending order
    sorted_df = results_df.sort_values(by='Predicted_Activity', ascending=False)
    
    # Save the sorted DataFrame to CSV
    predictions_file = 'gnn_predictions.csv'
    sorted_df.to_csv(predictions_file, index=False)
    
    print(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    main()