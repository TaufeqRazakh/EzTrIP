from ase.io import read
from pathlib import Path
import torch
import numpy as np
from eztrip.energy_corrector import *
from ase.calculators.emt import EMT

XYZ_PATH = Path(__file__).parent / 'test_inputs/raw/md.xyz'
training_structures = read(XYZ_PATH, index = ':')

def test_forces_from_input():    
    print(len(training_structures))

    forces = []
    for structure in training_structures:
        forces.append(torch.tensor(structure.get_forces()))
    print(forces[0].shape)
    print(f'The forces for all frames have length {len(forces)}')
    print(f'The forces for a single frame have length {forces[0].shape}')
    assert len(forces) > 0 , "Failed to read forces from input file"    

def test_ase_energy_shape():
    energies = []
    for structure in training_structures:
        energies.append(torch.tensor(structure.get_total_energy()))
    print(f'The energies for all frames have length {len(energies)}')

    all_species = get_all_species(training_structures)
    
    self_contributions = get_self_contributions(
            training_structures, all_species
        )

    train_energies = get_corrected_energies(
            training_structures, all_species, self_contributions
        )
    calc = EMT()
    ase_energy = []
    for structure in training_structures:
        structure.calc = calc
    ase_energy = np.array([structure.get_total_energy() for structure in training_structures])

    print(f'The energies from the self self_contributions are {train_energies}')
    print(f'The energies from the ase calculator is are {ase_energy}')
    assert train_energies.shape[0] > 0 , "Failed to read energies from input file"    # print(train_energies)

