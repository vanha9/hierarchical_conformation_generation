### 수정된 `codebook_data.py` + `collate_fn` 구현

import os
import pickle
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, txt_file, chain_dir, transform=None):
        with open(txt_file, 'r') as f:
            self.chain_ids = [line.strip() for line in f if line.strip()]
        self.chain_dir = chain_dir
        self.transform = transform


    def __len__(self):
        #return len(self.chain_ids)
        return 24000
        #return 32

    def __getitem__(self, idx):
        chain_id = self.chain_ids[idx]
        subdir = chain_id[1:3]
        path = os.path.join(self.chain_dir, subdir, f"{chain_id}.pkl")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        data.pop('ss', None)
        atom_positions = torch.tensor(data['atom_positions'], dtype=torch.float32)  # (L, 37, 3)
        L = atom_positions.shape[0]

        return atom_positions, L


def pad_collate_fn(batch, max_len=256):
    atom_positions_list, length_list = zip(*batch)  # 각각: list of (Lᵢ, 37, 3) / list of int
    B = len(atom_positions_list)
    #max_len = length_list[0]
    padded_atoms = torch.full((B, max_len, 37, 3), float('inf'))  # (B, 256, 37, 3)
    lengths = torch.tensor(length_list, dtype=torch.long)          # (B,)
    res_ids = torch.zeros((B, max_len), dtype=torch.long)         # (B, 256)

    for i, (atoms, L) in enumerate(zip(atom_positions_list, length_list)):
        padded_atoms[i, :L] = atoms
        res_ids[i, :L] = torch.arange(L)

    return padded_atoms, lengths, res_ids