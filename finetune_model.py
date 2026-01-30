import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool
from tqdm import tqdm  

from torch_geometric.loader import DataListLoader

from finetune_data import finetune_data_toy
from pretrain_model import PretrainModel

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)

class FinetuneModel(torch.nn.Module):
    def __init__(self, pretrain_model, hidden_dim, output_dim):
        super(FinetuneModel, self).__init__()
        self.pretrain = pretrain_model
        for params in self.pretrain.parameters():
            params.requires_grad = False
        self.output_layer = Linear(hidden_dim, output_dim) 

    def forward(self, batch, device):
        outputs = []
        for data in batch:
            data = data.to(device)
            h = self.pretrain.layer1(data.x, data.edge_index)
            mol_vec = global_add_pool(h, torch.zeros(data.x.shape[0], dtype=torch.long).to(device))
            
            out = self.output_layer(mol_vec)
            outputs.append(out)
            
        return torch.cat(outputs, dim=0)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


import json

def train(rank, world_size, dataset):
    set_seed(42 + rank)
    is_distributed = world_size > 1 and torch.cuda.is_available()

    with open('config.json', 'r') as f:
        config = json.load(f)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    pre_model = PretrainModel(input_dim=config['input_dim'],
                              hidden_dim=config['hidden_dim'],output_dim=1).to(device)

    checkpoint_path = os.path.join(os.getcwd(), 'best_pretrain_model')
    pre_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = FinetuneModel(pretrain_model=pre_model, hidden_dim=config['hidden_dim'], 
                          output_dim=1).to(device)

    if is_distributed:
        setup_ddp(rank, world_size)
        model = DDP(model, device_ids=[rank])

    loader = DataListLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    model.train()
    for epoch in tqdm(range(config['EPOCHS'])):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            y = model(batch, device)
            y_ground = torch.tensor([data.y for data in batch])
            y_ground = y_ground.view(-1, 1).to(torch.float32).to(device)
            loss = criterion(y, y_ground)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            
        print(f"Best Loss: {best_loss:.4f}")

    if is_distributed:
        cleanup_ddp()


import torch.multiprocessing as mp

if __name__ == "__main__":
    dataset = finetune_data_toy 
    
    gpus = torch.cuda.device_count()
    if gpus > 1:
        mp.spawn(train, args=(gpus, dataset), nprocs=gpus, join=True)
    else:
        print("No multiple GPUs found. Running on CPU")
        train(0, 1, dataset)




