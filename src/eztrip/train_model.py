import argparse
from ase.io import read
from pathlib import Path
import torch


def main(training_dir, xyz_filename):
    xyz_path = training_dir / xyz_filename
    training_structures = read(xyz_path, index=':')
    print(f"Number of training structures: {len(training_structures)}")

    forces = [torch.tensor(structure.get_forces()) for structure in training_structures]

    energies = [torch.tensor(structure.get_total_energy()) for structure in training_structures]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on molecular data.')
    parser.add_argument('training_dir', type=Path, help='Path to the directory containing the training data.')
    parser.add_argument('xyz_filename', type=str, help='Name of the .xyz or trajectory file.')
    args = parser.parse_args()
    
    main(args.training_dir, args.xyz_filename)
