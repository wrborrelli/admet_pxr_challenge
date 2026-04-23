import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def graph_features(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

class MoleculeDataset(Dataset):
    def __init__(self, df):
        self.smiles = np.stack(df.SMILES)
        self.emax = np.stack(df.Emax)
        self.targets = np.stack(df.pEC50)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles[idx])

        x = torch.tensor(
            [atom_features(a) for a in mol.GetAtoms()],
            dtype=torch.float
        )

        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_features(bond)

            edge_index += [[i, j], [j, i]]
            edge_attr += [bf, bf]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        g_feats = torch.tensor(graph_features(mol), dtype=torch.float).unsqueeze(0)
        emax = torch.tensor([[self.emax[idx]]], dtype=torch.float)
        g_feats = torch.cat([g_feats, emax], dim=1)

        y = torch.tensor(self.targets[idx], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_feats=g_feats)
    
def get_preds(model, loader):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr
            )

            preds.append(pred.reshape(-1).cpu())   # <-- FIX
            targets.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    return targets, preds
