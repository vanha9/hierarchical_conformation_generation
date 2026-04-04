import os
import pickle
import sys
from tqdm import tqdm
from esm.models.new_vqvae_copy import ProjectedEncoder 
import torch
import numpy as np
from Bio.PDB import PDBParser
import pandas as pd
import glob

def apply_special_tokens(token_tensor, lengths, bos_id, eos_id, pad_id):
    B, T = token_tensor.shape  # T = 256
    assert (lengths <= T).all(), "lengths must be <= T"

    total_len = T + 2
    device = lengths.device

    # 1. 전체 PAD로 채운 새 텐서 만들기
    result = torch.full((B, total_len), pad_id, device=device)

    # 2. BOS 삽입
    result[:, 0] = bos_id

    # 3. 토큰 복사: token_tensor[:, :Lᵢ] → result[:, 1:Lᵢ+1]
    idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T).to(device)  # (B, T)
    mask = idxs < lengths.unsqueeze(1)                      # (B, T)

    b_idx, t_idx = torch.nonzero(mask, as_tuple=True)
    result[b_idx, t_idx + 1] = token_tensor[b_idx, t_idx]
    eos_pos = lengths + 1  # EOS 위치는 L + 1
    result[torch.arange(B, device=device), eos_pos] = eos_id
    
    return result


atom_types = [
    "N", "CA", "C", "O", "CB",
    "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3",
    "OXT"
]
atom_order = {name: i for i, name in enumerate(atom_types)}

def pdb_to_atom37_tensor(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)

    # 첫 번째 model, 첫 번째 chain만 처리
    model = next(structure.get_models())
    chain = next(model.get_chains())

    residues = list(chain.get_residues())
    num_res = len(residues)

    # 0으로 초기화
    coords = np.zeros((num_res, 37, 3), dtype=np.float32)

    for res_idx, residue in enumerate(residues):
        for atom in residue.get_atoms():
            atom_name = atom.get_name().strip()
            if atom_name in atom_order:
                coords[res_idx, atom_order[atom_name], :] = atom.coord

    # torch.Tensor로 변환
    coords_tensor = torch.from_numpy(coords)  # shape: (num_residues, 37, 3)
    return coords_tensor

#device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).to(device)
encoder_ckpt = "/esm/esm/models/new_vqvae_huber/encoder_checkpoint_110.pth"
state_dict_encoder = torch.load(
    encoder_ckpt,
    map_location=device,
)
encoder.load_state_dict(state_dict_encoder['encoder_weight'], strict=True)
for param in encoder.parameters():
    param.requires_grad = False

pdb_files = glob.glob(os.path.join("/ped", "*.pdb"))
for GT_path in pdb_files:
        coordinates = pdb_to_atom37_tensor(f"{GT_path}")
        coordinates = torch.unsqueeze(coordinates, dim=0)
        print(f"coordinates shape : {coordinates.shape}")
        #coordinates = torch.tensor(data['atom_positions'], device=device)
        #coordinates = torch.unsqueeze(coordinates, dim=0)

        _,_,_, structure_tokens_large_t, structure_tokens_medium_t, structure_tokens_small_t, encoder_loss_large, encoder_loss_medium, encoder_loss_small = encoder.encode(
            coordinates
        )

        lengths = torch.tensor([structure_tokens_large_t.shape[1]], device=device)
        structure_tokens_large = apply_special_tokens(structure_tokens_large_t, lengths, 4096 + 2, 4096 + 1, 4096 + 3)
        structure_tokens_medium = apply_special_tokens(structure_tokens_medium_t, lengths, 512 + 2, 512 + 1, 512 + 3)
        structure_tokens_small = apply_special_tokens(structure_tokens_small_t, lengths, 32 + 2, 32 + 1, 32 + 3)

        structure_tokens_large = torch.squeeze(structure_tokens_large, dim=0)
        structure_tokens_medium = torch.squeeze(structure_tokens_medium, dim=0)
        structure_tokens_small = torch.squeeze(structure_tokens_small, dim=0)

        codebooks = {
            "large_codebook": structure_tokens_large.detach().cpu(),
            "medium_codebook": structure_tokens_medium.detach().cpu(),
            "small_codebook": structure_tokens_small.detach().cpu()
        }

        # .ptm 파일로 저장
        torch.save(codebooks, f"{GT_path[:-4]}.ptm")