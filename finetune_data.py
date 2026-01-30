import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
from utils.featurizers import Features

    
class FinetuneData(InMemoryDataset):
    def __init__(self,root,data_path):
        self.data_path = data_path
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['toy_finetune.csv']
    
    @property
    def processed_file_names(self):
        return ['finetune_data.pt']
    
    def process(self):
        feat = Features()
        df = pd.read_csv(self.data_path)
        df = df.dropna()

        smiles_names = ['Smiles','smiles','mol']
        df = pd.read_csv(self.raw_paths[0])
        df = df.dropna()
        smiles = set(smiles_names).intersection(list(df.columns))
        smiles_column_name = smiles.pop()

        data_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing Molecules'):
            smiles = row[smiles_column_name]
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
            x = feat.NodeFeatures(mol)
            edge_index = feat.EdgeIndex(mol)
            target_value = row['activity'] 

            y = torch.tensor([target_value], dtype=torch.float32).view(1, 1)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

finetune_data_toy = FinetuneData(root='data/finetune/', data_path='data/finetune/raw/toy_finetune.csv')
