import os
import pickle
import sys
from tqdm import tqdm
from esm.models.new_vqvae_copy import ProjectedDecoder 
import torch
from esm.utils.structure.protein_chain import ProteinChain
from esm.sdk.api import ESMProtein, ESMProteinTensor
import pickle

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']

def decode_seq(aatype):
    return ''.join([restypes_with_x[i] for i in aatype])

device = torch.device('cpu')
decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30).to(device)
decoder_ckpt = "/esm/esm/models/new_vqvae_huber/decoder_checkpoint_110.pth"
state_dict_decoder = torch.load(
    decoder_ckpt,
    map_location=device,
)
decoder.load_state_dict(state_dict_decoder, strict=False)
for param in decoder.parameters():
    param.requires_grad = False

pdb_code = '1u7i_B'
saved_gt_tokens = f'/{pdb_code}/gt_tokens.ptm'
coarse_tokens = f'/{pdb_code}/coarse_saved_tokens.pt'

gt_tokens = torch.load(saved_gt_tokens, map_location=device)

lengths = torch.tensor([gt_tokens['large_codebook'].shape[0] - 2])

tokens_large = gt_tokens['large_codebook'].unsqueeze(dim=0)
tokens_mid = gt_tokens['medium_codebook'].unsqueeze(dim=0)
tokens_small = gt_tokens['small_codebook'].unsqueeze(dim=0)

output = decoder.decode(tokens_large, tokens_mid, tokens_small, lengths)

large_pred = output["bb_pred_large"].squeeze(dim=0)[1:-1].detach().cpu()
mid_pred = output["bb_pred_medium"].squeeze(dim=0)[1:-1].detach().cpu()
small_pred = output["bb_pred_small"].squeeze(dim=0)[1:-1].detach().cpu()

with open(file='/1u7i_B/1u7i_B.pkl', mode='rb') as f:
    data_dict=pickle.load(f)
sequence = decode_seq(data_dict['aatype'])

large_chain = ProteinChain.from_backbone_atom_coordinates(large_pred, sequence=sequence)
large_chain = large_chain.infer_oxygen()
large_coord = torch.tensor(large_chain.atom37_positions)
fine_prot = ESMProtein(sequence=sequence, coordinates=large_coord)

mid_chain = ProteinChain.from_backbone_atom_coordinates(mid_pred, sequence=sequence)
mid_chain = mid_chain.infer_oxygen()
mid_coord = torch.tensor(mid_chain.atom37_positions)
mid_prot = ESMProtein(sequence=sequence, coordinates=mid_coord)

small_chain = ProteinChain.from_backbone_atom_coordinates(small_pred, sequence=sequence)
small_chain = small_chain.infer_oxygen()
small_coord = torch.tensor(small_chain.atom37_positions)
coarse_prot = ESMProtein(sequence=sequence, coordinates=small_coord)

fine_prot.to_pdb(f'/{pdb_code}/fine_gt.pdb')
mid_prot.to_pdb(f'/{pdb_code}/mid_gt.pdb')
coarse_prot.to_pdb(f'/{pdb_code}/coarse_gt.pdb')
print(output["bb_pred_large"].shape)
print(output["bb_pred_medium"].shape)
print(output["bb_pred_small"].shape)
