import os
import pickle
import sys
from tqdm import tqdm
from esm.models.new_vqvae_copy import ProjectedEncoder 
import torch

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).to(device)
encoder_ckpt = "/esm/esm/models/new_vqvae_huber/encoder_checkpoint_110.pth"
state_dict_encoder = torch.load(
    encoder_ckpt,
    map_location=device,
)
encoder.load_state_dict(state_dict_encoder['encoder_weight'], strict=True)
for param in encoder.parameters():
    param.requires_grad = False


txt_file="/data/pdb_data/processed_chains/dssp_success_1000.txt"
chain_dir="/data/pdb_data/processed_chains"
first_processed_dir="/data/pdb_data/processed_chains"

def decode_seq(aatype):
    return ''.join([restypes_with_x[i] for i in aatype])
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']


with open(txt_file, 'r') as f:
    chain_ids = [line.strip() for line in f if line.strip()]
print(f"data num : {len(chain_ids)}")

for chain_id in tqdm(chain_ids[60000:120000]):
        
    if os.path.exists(f"/data/structure_preprocess/{chain_id[1:3]}/{chain_id}.ptm"):
        continue
    path = os.path.join(chain_dir, chain_id[1:3], f"{chain_id}.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data.pop('ss', None)
    coordinates = torch.tensor(data['atom_positions'], device=device)
    coordinates = torch.unsqueeze(coordinates, dim=0)

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

    subfolder = f"/data/structure_preprocess/{chain_id[1:3]}"
    os.makedirs(subfolder, exist_ok=True)
    # .ptm 파일로 저장
    torch.save(codebooks, f"/data/structure_preprocess/{chain_id[1:3]}/{chain_id}.ptm")