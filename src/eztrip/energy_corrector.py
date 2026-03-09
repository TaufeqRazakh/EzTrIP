import numpy as np
import ase
from sklearn.linear_model import Ridge

def get_all_species(structures):
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)
    all_species = np.sort(np.unique(all_species))
    return all_species

def get_self_contributions(train_structures, all_species):
    train_energies = np.array(
        [structure.get_total_energy() for structure in train_structures]
    )
    train_c_feat = get_compositional_features(train_structures, all_species)
    rgr = Ridge(alpha=1e-10, fit_intercept=False)
    rgr.fit(train_c_feat, train_energies)
    return rgr.coef_

def get_compositional_features(structures, all_species):
    result = np.zeros([len(structures), len(all_species)])
    for i, structure in enumerate(structures):
        species_now = structure.get_atomic_numbers()
        for j, specie in enumerate(all_species):
            num = np.sum(species_now == specie)
            result[i, j] = num
    return result

def get_corrected_energies(structures, all_species, self_contributions):
    energies = np.array([structure.get_total_energy() for structure in structures])

    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(
            np.dot(compositional_features[i], self_contributions)
        )
    self_contributions_energies = np.array(self_contributions_energies)
    return energies - self_contributions_energies

__all__ = ["get_all_species", "get_corrected_energies", "get_self_contributions",
           "get_compositional_features"]