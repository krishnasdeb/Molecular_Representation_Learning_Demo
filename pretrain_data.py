import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
from utils.featurizers import Features

    
class PretrainData(InMemoryDataset):
    def __init__(self,root,data_path):
        self.data_path = data_path
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['toy_pretrain.csv']
    
    @property
    def processed_file_names(self):
        return ['pretrain_data.pt']
    
    def process(self):
        feat = Features()
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=['smiles'])

        data_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing Molecules'):
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
            x = feat.NodeFeatures(mol)
            edge_index = feat.EdgeIndex(mol)
            y = feat.training_task(mol)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

pretrain_data_toy = PretrainData(root='data/pretrain/', data_path='data/pretrain/raw/toy_pretrain.csv')
