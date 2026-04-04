import os
import pickle
import sys
from tqdm import tqdm
from esm.models.vqvae import * 
import torch
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils import encoding

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

def kabsch_align_torch(P: torch.Tensor, Q: torch.Tensor):
    assert P.shape == Q.shape, "P and Q must have the same shape"
    P_flat = P.view(-1, 3)
    Q_flat = Q.view(-1, 3)

    # 중심 정렬
    P_mean = P_flat.mean(dim=0, keepdim=True)
    Q_mean = Q_flat.mean(dim=0, keepdim=True)
    P_cent = P_flat - P_mean
    Q_cent = Q_flat - Q_mean

    # 공분산 행렬
    C = torch.matmul(P_cent.T, Q_cent)

    # SVD
    V, S, Wt = torch.linalg.svd(C)
    d = torch.sign(torch.linalg.det(torch.matmul(V, Wt)))
    D = torch.diag(torch.tensor([1., 1., d], device=P.device))

    # 회전 행렬
    U = torch.matmul(torch.matmul(V, D), Wt)

    # P에 회전 적용
    P_aligned = torch.matmul(P_cent, U)

    return P_aligned, Q_cent

def compute_rmsd_aligned_torch(X_hat, X):
    """
    X_hat, X: (B, 256, 3, 3)
    lengths: (B,)
    Returns: 평균 RMSD over batch
    """
    rmsds = []

    P = X_hat   # (L, 3, 3)
    Q = X

        # flatten to (L*3, 3)
    P = P.reshape(-1, 3)
    Q = Q.reshape(-1, 3)

    P_aligned, Q_cent = kabsch_align_torch(P, Q)
    diff = P_aligned - Q_cent
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
    rmsds.append(rmsd)

    return torch.stack(rmsds).mean() if rmsds else torch.tensor(0.0, device=X.device)

#모델들 load
device = torch.device("cpu")
encoder_ckpt="/data/esm3_checkpoint/esm3_structure_encoder_v0.pth"
decoder_ckpt="/data/esm3_checkpoint/esm3_structure_decoder_v0.pth"

encoder = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).to(device)
state_dict_encoder = torch.load(encoder_ckpt, map_location=device)
encoder.load_state_dict(state_dict_encoder, strict=False)

decoder = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).to(device)
state_dict_decoder = torch.load(decoder_ckpt, map_location=device)
decoder.load_state_dict(state_dict_decoder,  strict=False)

#데이터들 읽어오는 경로
txt_file="/data/pdb_data/processed_chains/dssp_success_short_256.txt"
chain_dir="/data/pdb_data/processed_chains"
with open(txt_file, 'r') as f:
    chain_ids = [line.strip() for line in f if line.strip()]
chain_ids = chain_ids[:24000]

#읽어오기
for chain_id in tqdm(chain_ids[:10]):
    subdir = chain_id[1:3]
    path = os.path.join(chain_dir, subdir, f"{chain_id}.pkl")
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)

    coordinates = torch.tensor(data_dict['atom_positions']).to(device)
    #sequence = torch.tensor(decode_seq(data_dict['aatype'])).to(device)

    gt_coords = coordinates[..., :3, :]
    print(coordinates.shape)
    _, structure_tokens, _ = encoder.encode(
        coordinates.unsqueeze(dim=0)
    )

    structure_tokens = torch.cat(
        [torch.LongTensor([4098]), 
        structure_tokens.squeeze(), 
        torch.LongTensor([4097])]
    ).unsqueeze(dim=0)

    decoder_output = decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()
    gt_coords = gt_coords.double()
    bb_coords = bb_coords.double()

    print(f"gt_coords shape : {gt_coords.shape}")
    print(f"bb_coords shape : {bb_coords.shape}")
    rmsd = compute_rmsd_aligned_torch(gt_coords, bb_coords)
    print(f"rmsd : {rmsd}")
