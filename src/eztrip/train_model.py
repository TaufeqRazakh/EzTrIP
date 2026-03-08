import argparse
from ase.io import read
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader


def main(xyz_filename):
    training_structures = read(xyz_filename, index=':')
    print(f"Number of training structures: {len(training_structures)}")

    forces = [torch.tensor(structure.get_forces()) for structure in training_structures]

    energies = [torch.tensor(structure.get_total_energy()) for structure in training_structures]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on molecular data.')
    parser.add_argument('-d', '--training_dir', type=Path, 
                        help='Path to the directory containing the training data.')
    parser.add_argument('-f', '--filename', type=str, 
                        help='Name of the .xyz or trajectory file.')
    args = parser.parse_args()
    if args.training_dir:
        posix_paths = args.training_dir.glob('*.xyz')
        for filename in sorted(posix_paths):
            main(filename)
    else:
        main(args.filename)
