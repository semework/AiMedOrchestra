import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from transformers import pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Simple GIN model for demonstration
class GIN(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=1):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x

class DrugDiscoveryAgent:
    def __init__(self):
        # Initialize the GNN model (for demo purposes, random weights)
        self.model = GIN()
        self.model.eval()
        # Initialize text generation for drug naming/explanation
        self.generator = pipeline('text-generation', model='gpt2', max_length=50)
        # Mock dataset of molecules as SMILES strings
        self.molecules = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl',  # Diazepam
            'CC1=CC=CC=C1',  # Toluene (placeholder)
            'C1=CC=C(C=C1)C=O',  # Benzaldehyde
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            # Add more molecules here...
        ]

    def mol_to_graph_data(self, smiles):
        # Convert SMILES to PyG Data object (basic example)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Atom features: here just atomic numbers normalized (simple demo)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([atom.GetAtomicNum() / 100.0])  # simple normalized atomic number

        x = torch.tensor(atom_features, dtype=torch.float)
        # Build edge index (bonds)
        edge_index = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data

    def suggest(self, condition):
        # Convert molecules to graph data
        data_list = []
        valid_smiles = []
        for smi in self.molecules:
            data = self.mol_to_graph_data(smi)
            if data is not None:
                data_list.append(data)
                valid_smiles.append(smi)
        # Create batch
        batch = DataLoader(data_list, batch_size=len(data_list))
        batch_data = next(iter(batch))

        # Predict scores (random weights, demo only)
        with torch.no_grad():
            scores = self.model(batch_data.x, batch_data.edge_index, batch_data.batch).squeeze()
        # Sort molecules by score descending
        sorted_idx = torch.argsort(scores, descending=True)
        top_smiles = [valid_smiles[i] for i in sorted_idx[:3]]  # top 3 for demo

        # Use LLM to generate drug names/descriptions
        condition_prompt = f"Suggest top 3 drug candidates for treating {condition}: "
        llm_input = condition_prompt + ', '.join(top_smiles)
        llm_output = self.generator(llm_input, max_length=100, num_return_sequences=1)[0]['generated_text']

        return {
            'top_molecules': top_smiles,
            'description': llm_output[len(condition_prompt):].strip()
        }


# Example usage:
if __name__ == "__main__":
    agent = DrugDiscoveryAgent()
    condition = "Type 2 Diabetes"
    suggestions = agent.suggest(condition)
    print("Top candidate molecules (SMILES):", suggestions['top_molecules'])
    print("LLM description:", suggestions['description'])
