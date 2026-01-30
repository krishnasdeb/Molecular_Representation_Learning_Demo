import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

class Features:
    def NodeFeatures(self, molecule):
        atomic_number = []
        atomic_aromaticity = []
        atomic_degree = []
        for atom in molecule.GetAtoms():
            atomic_number.append(atom.GetAtomicNum())
            atomic_aromaticity.append(int(atom.GetIsAromatic()))
            atomic_degree.append(atom.GetDegree())

        final_features = np.array([atomic_aromaticity, atomic_degree, atomic_number]).T
        return torch.from_numpy(final_features)
    
    def EdgeIndex(self,molecule):
        edges = []
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def training_task(self, molecule):
        task = []
        wt = Descriptors.MolWt(molecule)
        task.append(wt)
        return task
    