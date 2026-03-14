import glob
import os
import os.path as osp
from typing import Callable, List, Optional
from ase.calculators.emt import EMT
import torch
from pathlib import Path
from ase.io import read
import numpy as np
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch_geometric.io import read_off


class MDFrames(InMemoryDataset):
    r"""Dataset of XYZ frames.

    .. note::

        Data objects holds xyz co-ordinates

    Args:
        root (str): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    """
    # url = 'https://github.com/Yannick-S/geometric_shapes/raw/master/raw.zip'

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        # path = self.processed_paths[0] if train else self.processed_paths[1]
        # self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name]

    @property
    def processed_file_names(self) -> List[str]:
        name = f"{Path(self.name).stem}.pt"
        return [name]

    def download(self) -> None:
        ...

    def process(self) -> None:
        atoms_path = Path(self.raw_dir) / self.name
        atoms_data = read(atoms_path, ':')

        data_list = []
        calc = EMT()
        for frame in atoms_data:
            frame.calc = calc
        energies = np.array([frame.get_total_energy() for frame in atoms_data])

        for i, atoms_frame in enumerate(atoms_data):
            pos = torch.tensor(atoms_frame.get_positions(), dtype=torch.float)
            z = torch.tensor(atoms_frame.get_atomic_numbers(), dtype=torch.long)
            f = torch.tensor(atoms_frame.get_forces(), dtype=torch.float)
            
            energy = energies[i]
            if not isinstance(energy, torch.Tensor):
                energy = torch.tensor([energy], dtype=torch.float)
            elif energy.dim() == 0:
                energy = energy.unsqueeze(0)

            data = Data(pos=pos, z=z, f=f, y=energy)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

__all__ = ["MDFrames"]