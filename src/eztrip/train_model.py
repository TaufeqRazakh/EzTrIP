import argparse
from ase.io import read
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import time
from eztrip.frames import MDFrames
import yaml
from pathlib import Path

def train_model(config):
    data_path = Path(config['data']['data_path'])
    print(f'root is {data_path.parents[1]} and filename is {data_path.name}')
    atoms_dataset = MDFrames(root=str(data_path.parents[1]),
                              name=data_path.name)
    
#     
#     

#     Data()
#     timestamp = time.strftime("%Y%m%d-%H%M")
#     species_path = Path('.') / timestamp
#     species_path.mkdir(exist_ok=True)
#     np.save(species_path/'all_species', all_species)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on molecular data.')
    parser.add_argument('-c', '--config_file', type=Path, 
                        help='Path to the config file for training')
    args = parser.parse_args()
    yaml_path = args.config_file
    with yaml_path.open(mode='r') as f:
        config = yaml.safe_load(f)
        train_model(config)
