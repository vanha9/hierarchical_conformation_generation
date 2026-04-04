import os
from pathlib import Path
import pickle
import sys
from tqdm import tqdm
from esm.models.new_vqvae_copy import ProjectedDecoder 
import torch
from esm.utils.structure.affine3d import Affine3D 
from esm.utils import residue_constants
from Bio.PDB import PDBParser
from esm.sdk.api import ESMProtein, ESMProteinTensor
import tempfile
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from esm.utils.structure.protein_chain import ProteinChain

def merge_pdbfiles(input: Path, save_to: Path, verbose=True):
    """ordered merging process of pdbs"""
    if isinstance(input, Path):
        pdb_files = [f for f in input.iterdir() if f.suffix == '.pdb']
    elif isinstance(input, list):
        pdb_files = input
    else:
        raise ValueError(f"Unrecognized input type: {type(input)}")

    assert len(pdb_files) > 0
        
    save_to.parent.mkdir(parents=True, exist_ok=True)
    
    model_number = 0
    pdb_lines = []
    if verbose: 
        _iter = tqdm(pdb_files, desc='Merging PDBs')
    else:
        _iter = pdb_files
    
    for pdb_file in _iter:
        with open(pdb_file, 'r') as pdb:
            lines = pdb.readlines()
        single_model = True
        
        for line in lines: 
            if line.startswith('MODEL') or line.startswith('ENDMDL'):
                single_model = False
                break
        
        if single_model: # single model
            model_number += 1
            pdb_lines.append(f"MODEL     {model_number}")
            for line in lines: 
                if line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
            pdb_lines.append("ENDMDL")
        else:        # multiple models
            for line in lines:
                if line.startswith('MODEL'):
                    model_number += 1
                    if model_number > 1:
                        pdb_lines.append("ENDMDL")
                    pdb_lines.append(f"MODEL     {model_number}")
                elif line.startswith('END'):
                    continue
                elif line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')
    pdb_lines = [_line.ljust(80) for _line in pdb_lines]
    pdb_str = '\n'.join(pdb_lines) + '\n'
    with open(save_to, 'w') as fo:
        fo.write(pdb_str)
    if verbose:
        print(f"Merged {len(pdb_files)} PDB files into {save_to} with {model_number} models.")

def get_sequence_string_from_pdb(pdb_file: str, chain_id: str = None) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                ppb = PPBuilder()
                peptides = ppb.build_peptides(chain)
                if peptides:
                    sequence = peptides[0].get_sequence()
                    return str(sequence)
                else:
                    return ""  # No peptide found
        break  # 첫 번째 모델만

    return ""

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

device = torch.device('cpu')
decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30).to(device)
decoder_ckpt = "/esm/esm/models/new_vqvae_huber/decoder_checkpoint_110.pth"
state_dict_decoder = torch.load(
    decoder_ckpt,
    map_location=device,
)
decoder.load_state_dict(state_dict_decoder['decode_proj'], strict=True)
for param in decoder.parameters():
    param.requires_grad = False

saved_tokens = "/esmdiff_var/slm/models/esmdiff_residual_scaling/bpti/bpti_token_01.pth"
tokens_bpti = torch.load(saved_tokens, map_location=device)
#structure_token = tokens_bpti["fine_structure_tokens"]
structure_token_coarse = tokens_bpti[(0, 'result')]
structure_token_mid = tokens_bpti[(1, 'result')] - 32
structure_token_fine = tokens_bpti[(2, 'result')] - 32 - 512

lengths = torch.tensor([structure_token_fine.shape[1] - 2] * structure_token_fine.shape[0])

structure_token_fine = apply_special_tokens(structure_token_fine[:, 1:-1], lengths, 4096 + 2, 4096 + 1, 4096 + 3)
structure_token_mid = apply_special_tokens(structure_token_mid[:, 1:-1], lengths, 512 + 2, 512 + 1, 512 + 3)
structure_token_coarse = apply_special_tokens(structure_token_coarse[:, 1:-1], lengths, 32 + 2, 32 + 1, 32 + 3)


output = decoder.decode(structure_token_fine, structure_token_mid, structure_token_coarse, lengths)

#sequence = get_sequence_string_from_pdb("/md_result/villin/2RJY_Unfold_927autopsf.pdb")
#sequence = "LETFPLDVLVNTAAEDLPRGVDPSRKENLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
sequence = get_sequence_string_from_pdb("/esmdiff_var/data/targets/bpti/bpti.pdb")

ensemble_bb_coords = []
saved = []
with tempfile.TemporaryDirectory() as tmpdirname:
    for i, pred_large in enumerate(output['bb_pred_large']):
        base = "predict_bpti_1sfine" + '.' + f"{i}.pdb"
        tmp = Path(tmpdirname) / base

        bb_coords: torch.Tensor = pred_large.unsqueeze(dim=0)[
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens
        bb_coords = bb_coords.detach().cpu()
        chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
        chain = chain.infer_oxygen()

        protein_predicted = ESMProtein(
                sequence=sequence,
                secondary_structure=None,
                sasa=None,  # type: ignore
                function_annotations=None,
                coordinates=torch.tensor(chain.atom37_positions),
                plddt=None,
                ptm=None,
                potential_sequence_of_concern=None,
            )

        protein_predicted.to_pdb(tmp)
        ensemble_bb_coords.append(bb_coords.unsqueeze(dim=0))
        saved.append(tmp)
    merge_pdbfiles(saved, Path("/esmdiff_var/slm/models/esmdiff_residual_scaling/bpti/step25_eps1e-05_N100_20251105-235924/bpti_fine.pdb"), verbose=False)

ensemble_bb_coords = []
saved = []
with tempfile.TemporaryDirectory() as tmpdirname:
    for i, pred_medium in enumerate(output['bb_pred_medium']):
        base = "predict_bpti_1smedium" + '.' + f"{i}.pdb"
        tmp = Path(tmpdirname) / base

        bb_coords: torch.Tensor = pred_medium.unsqueeze(dim=0)[
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens
        bb_coords = bb_coords.detach().cpu()
        chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
        chain = chain.infer_oxygen()

        protein_predicted = ESMProtein(
                sequence=sequence,
                secondary_structure=None,
                sasa=None,  # type: ignore
                function_annotations=None,
                coordinates=torch.tensor(chain.atom37_positions),
                plddt=None,
                ptm=None,
                potential_sequence_of_concern=None,
            )

        protein_predicted.to_pdb(tmp)
        ensemble_bb_coords.append(bb_coords.unsqueeze(dim=0))
        saved.append(tmp)
    merge_pdbfiles(saved, Path("/esmdiff_var/slm/models/esmdiff_residual_scaling/bpti/step25_eps1e-05_N100_20251105-235924/bpti_mid.pdb"), verbose=False)

ensemble_bb_coords = []
saved = []
with tempfile.TemporaryDirectory() as tmpdirname:
    for i, pred_coarse in enumerate(output['bb_pred_small']):
        base = "predict_bpti_1scoarse" + '.' + f"{i}.pdb"
        tmp = Path(tmpdirname) / base

        bb_coords: torch.Tensor = pred_coarse.unsqueeze(dim=0)[
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens
        bb_coords = bb_coords.detach().cpu()
        chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
        chain = chain.infer_oxygen()

        protein_predicted = ESMProtein(
                sequence=sequence,
                secondary_structure=None,
                sasa=None,  # type: ignore
                function_annotations=None,
                coordinates=torch.tensor(chain.atom37_positions),
                plddt=None,
                ptm=None,
                potential_sequence_of_concern=None,
            )

        protein_predicted.to_pdb(tmp)
        ensemble_bb_coords.append(bb_coords.unsqueeze(dim=0))
        saved.append(tmp)
    merge_pdbfiles(saved, Path("/esmdiff_var/slm/models/esmdiff_residual_scaling/bpti/step25_eps1e-05_N100_20251105-235924/bpti_coarse.pdb"), verbose=False)
