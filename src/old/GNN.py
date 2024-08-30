import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

print(torch.__version__)
dml = torch_directml.device()

def prepare_data(file):
    smiles_data = pd.read_csv(file)
    features_list = []
    edge_index_list = []
    labels = []

    # Convert string labels to numerical values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(smiles_data['Active'])

    for i in range(len(smiles_data['Std_Smiles'])):
        smile = smiles_data['Std_Smiles'].iloc[i]
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            features, edge_index = mol_to_features_and_edge_index(mol)
            features_list.append(features)
            edge_index_list.append(edge_index)
            labels.append(encoded_labels[i])

    return features_list, edge_index_list, labels, label_encoder

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

def train(model, train_features, train_edge_indices, train_labels, optimizer, device):
    model.train()
    total_loss = 0
    for features, edge_index, label in zip(train_features, train_edge_indices, train_labels):
        features, edge_index = features.to(device), edge_index.to(device)
        label = torch.tensor([label], dtype=torch.float).to(device)
        optimizer.zero_grad()
        output = model(features, edge_index)
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_labels)

def evaluate(model, features, edge_indices, labels, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, edge_index, label in zip(features, edge_indices, labels):
            features, edge_index = features.to(device), edge_index.to(device)
            output = model(features, edge_index)
            predictions.append(output.item())
    return roc_auc_score(labels, predictions), f1_score(labels, [1 if p > 0.5 else 0 for p in predictions])

def main():
    # Prepare Data
    features_list, edge_index_list, labels, label_encoder = prepare_data('data/cdk8-smiles.csv')
    
    # Split the data
    train_features, test_features, train_edge_indices, test_edge_indices, train_labels, test_labels = train_test_split(
        features_list, edge_index_list, labels, test_size=0.2, random_state=42)
    
    # Initialize model
    device = torch_directml.device()
    model = GNN(num_features=4).to(device)  # 4 features as defined in mol_to_features_and_edge_index
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    
    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, train_features, train_edge_indices, train_labels, optimizer, device)
        train_auc, train_precision = evaluate(model, train_features, train_edge_indices, train_labels, device)
        test_auc, test_precision = evaluate(model, test_features, test_edge_indices, test_labels, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Train Precision: {train_precision:.4f}, '
              f'Test AUC: {test_auc:.4f}, Test Precision: {test_precision:.4f}')
    
    # Final evaluation
    test_auc, test_precision = evaluate(model, test_features, test_edge_indices, test_labels, device)
    print(f'\nFinal Test AUC: {test_auc:.4f}, Final Test Precision: {test_precision:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'gnn_bioactivity_model.pth')

if __name__ == "__main__":
    main()