import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

#from models.vqvae import StructureTokenEncoder, StructureTokenDecoder
from esm.models.vqvae import *

from esm.models.codebook_data_copy import ProteinDataset
from esm.models.codebook_data_copy import pad_collate_fn
from esm.tokenization.structure_tokenizer import StructureTokenizer

from esm.layers.blocks import UnifiedTransformerBlock
from esm.layers.codebook import EMACodebook
from esm.layers.structure_proj import Dim6RotStructureHead
from esm.layers.transformer_stack import TransformerStack
from esm.utils.constants import esm3 as C
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os 

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def apply_special_tokens(token_tensor, lengths, bos_id, eos_id, pad_id):
    B, T = token_tensor.shape  # T = 256
    assert (lengths <= T).all(), "lengths must be <= T"

    total_len = T + 2
    device = lengths.device

    # Initialize the output tensor with PAD tokens.
    result = torch.full((B, total_len), pad_id, device=device)

    # Insert BOS at the first position.
    result[:, 0] = bos_id

    # Copy each valid token span after BOS.
    idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T).to(device)  # (B, T)
    mask = idxs < lengths.unsqueeze(1)                      # (B, T)

    b_idx, t_idx = torch.nonzero(mask, as_tuple=True)
    result[b_idx, t_idx + 1] = token_tensor[b_idx, t_idx]
    eos_pos = lengths + 1  # EOS is placed at L + 1.
    result[torch.arange(B, device=device), eos_pos] = eos_id
    
    return result

def kabsch_align_torch(P: torch.Tensor, Q: torch.Tensor):
    assert P.shape == Q.shape, "P and Q must have the same shape"
    P_flat = P.view(-1, 3)
    Q_flat = Q.view(-1, 3)

    # Center both point clouds.
    P_mean = P_flat.mean(dim=0, keepdim=True)
    Q_mean = Q_flat.mean(dim=0, keepdim=True)
    P_cent = P_flat - P_mean
    Q_cent = Q_flat - Q_mean

    # Covariance matrix.
    C = torch.matmul(P_cent.T, Q_cent)

    # SVD
    V, S, Wt = torch.linalg.svd(C)
    d = torch.sign(torch.linalg.det(torch.matmul(V, Wt)))
    D = torch.diag(torch.tensor([1., 1., d], device=P.device))

    # Rotation matrix.
    U = torch.matmul(torch.matmul(V, D), Wt)

    # Apply rotation to P.
    P_aligned = torch.matmul(P_cent, U)

    return P_aligned, Q_cent

def compute_rmsd_aligned_torch(X_hat, X, lengths):
    """
    X_hat, X: (B, 256, 3, 3)
    lengths: (B,)
    Returns the mean RMSD over the batch.
    """
    B = X.shape[0]
    rmsds = []

    for b in range(B):
        L = lengths[b]
        if L < 1:
            continue

        P = X_hat[b, 1:L+1]  # (L, 3, 3)
        Q = X[b, :L]

        # flatten to (L*3, 3)
        P = P.reshape(-1, 3)
        Q = Q.reshape(-1, 3)

        P_aligned, Q_cent = kabsch_align_torch(P, Q)
        diff = P_aligned - Q_cent
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
        rmsds.append(rmsd)

    return torch.stack(rmsds).mean() if rmsds else torch.tensor(0.0, device=X.device)

def backbone_distance_loss(X_hat, X, lengths):
    """
    X_hat, X: (B, 256, 3, 3)
    lengths: (B,) valid sequence length for each batch item
    """
    B = X.shape[0]
    losses = []

    for b in range(B):
        L = lengths[b]
        Z_hat = X_hat[b, 1:L+1].reshape(-1, 3)  # (L*3, 3)
        Z = X[b, :L].reshape(-1, 3)

        D_pred = torch.cdist(Z_hat, Z_hat, p=2).pow(2)
        D_true = torch.cdist(Z, Z, p=2).pow(2)

        E = (D_pred - D_true).pow(2)
        E = torch.minimum(E, torch.tensor(25.0, device=E.device))
        losses.append(E.mean())

    return torch.stack(losses).mean()

def backbone_distance_loss_Huber(X_hat, X, lengths):
    B = X.shape[0]
    losses = []

    for b in range(B):
        L = lengths[b]
        Z_hat = X_hat[b, 1:L+1].reshape(-1, 3)
        Z = X[b, :L].reshape(-1, 3)

        D_pred = torch.cdist(Z_hat, Z_hat, p=2)
        D_true = torch.cdist(Z, Z, p=2)

        E = (D_pred - D_true).pow(2)

        # Smooth clipping using Huber-like adjustment
        delta = 5.0
        E = torch.where(E < delta**2, E, 2*delta*torch.sqrt(E + 1e-6) - delta**2)

        losses.append(E.mean())

    return torch.stack(losses).mean() * 0.02

def compute_vectors(X):
    """
    X: (L, 3, 3)
    Output: (L-1, 6, 3) - vector set
    """
    N, CA, C = X[:, 0], X[:, 1], X[:, 2]
    C_next = torch.roll(N, shifts=-1, dims=0)

    v1 = CA - N
    v2 = C - CA
    v3 = C_next - C
    v4 = -torch.cross(v1, v2, dim=-1)
    v5 = torch.cross(v3, v1, dim=-1)
    v6 = torch.cross(v2, v3, dim=-1)

    return torch.stack([v1, v2, v3, v4, v5, v6], dim=1)  # (L-1, 6, 3)

def backbone_direction_loss(X_hat, X, lengths):
    B = X.shape[0]
    losses = []

    for b in range(B):
        L = lengths[b]
        if L < 2:  # Vector operations require at least two residues.
            continue

        V_hat = compute_vectors(X_hat[b, 1:L+1])  # (L-1, 6, 3)
        V = compute_vectors(X[b, :L])          # (L-1, 6, 3)

        V_hat = V_hat.reshape(-1, 3)  # ((L-1)*6, 3)
        V = V.reshape(-1, 3)

        D_pred = V_hat @ V_hat.T  # ((L-1)*6, (L-1)*6)
        D_true = V @ V.T

        E = (D_pred - D_true).pow(2)
        E = torch.minimum(E, torch.tensor(20.0, device=E.device))
        losses.append(E.mean())

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=X.device)

class ProjectedEncoder(StructureTokenEncoder):
    def __init__(self, *args, mid_size=512, small_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.medium_codebook = EMACodebook(mid_size, kwargs['d_out'])
        self.small_codebook = EMACodebook(small_size, kwargs['d_out'])

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)


        #if attention_mask is None:
        #    attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        #attention_mask = attention_mask.bool()
        if attention_mask is None:
            attention_mask = affine_mask.bool()
        # print("============affine_mask============")
        # print(affine_mask.shape)
        # print(affine_mask[0])
        # print("============attention mask============")
        # print(attention_mask.shape)
        # print(attention_mask[0])
        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)


        z_q_large, min_encoding_indices_large, encoder_loss_large = self.codebook(z)
        z_q_medium, min_encoding_indices_medium, encoder_loss_medium = self.medium_codebook(z)
        z_q_small, min_encoding_indices_small, encoder_loss_small = self.small_codebook(z)

        #return z_q, min_encoding_indices
        return z_q_large, z_q_medium, z_q_small, min_encoding_indices_large, min_encoding_indices_medium, min_encoding_indices_small, encoder_loss_large, encoder_loss_medium, encoder_loss_small

class ProjectedDecoder(StructureTokenDecoder):
    def __init__(self, *args, mid_size=512, small_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.medium_embed = nn.Embedding(
            mid_size + len(self.special_tokens), kwargs['d_model']
        )
        self.small_embed = nn.Embedding(
            small_size + len(self.special_tokens), kwargs['d_model']
        )

    def decode(
        self,
        structure_tokens_large: torch.Tensor,
        structure_tokens_medium: torch.Tensor,
        structure_tokens_small: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ):
        B = lengths.shape[0]
        idxs = torch.arange(structure_tokens_large.shape[1], device=lengths.device).unsqueeze(0)  # shape: (1, 256)
        attention_mask = idxs < (lengths + 2).unsqueeze(1)
        #if attention_mask is None:
        #    attention_mask = torch.ones_like(structure_tokens_large, dtype=torch.bool)

        #attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens_large, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens_large, dtype=torch.int64)

        # check that BOS and EOS are set correctly
        assert (
            structure_tokens_large[:, 0].eq(self.special_tokens["BOS"]).all()
        ), "First token in structure_tokens must be BOS token"
        assert (
            structure_tokens_large[
                torch.arange(structure_tokens_large.shape[0]), attention_mask.sum(1) - 1
            ]
            .eq(self.special_tokens["EOS"])
            .all()
        ), "Last token in structure_tokens must be EOS token"
        assert (
            (structure_tokens_large < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        # Build broadcastable row/column indices.
        L = structure_tokens_large.shape[1]
        row_idx = torch.arange(L, device=lengths.device).view(1, L, 1)  # [1, L, 1]
        col_idx = torch.arange(L, device=lengths.device).view(1, 1, L)  # [1, 1, L]

        lengths_exp = lengths.view(B, 1, 1) + 2  # [B, 1, 1]

        mask = (row_idx >= lengths_exp) | (col_idx >= lengths_exp)  # [B, L, L]

        attn_mask = torch.zeros((B, L, L), device=lengths.device)
        attn_mask = attn_mask.masked_fill(mask, float('-inf'))
        x_large = self.embed(structure_tokens_large)
        x_medium = self.medium_embed(structure_tokens_medium)
        x_small = self.small_embed(structure_tokens_small)
        # !!! NOTE: Attention mask is actually unused here so watch out
        x_large, _, _ = self.decoder_stack.forward(
            x_large, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id, attn_mask = attn_mask
        )
        x_medium, _, _ = self.decoder_stack.forward(
            x_medium, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id, attn_mask = attn_mask
        )
        x_small, _, _ = self.decoder_stack.forward(
            x_small, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id, attn_mask = attn_mask
        )


        tensor7_affine_large, bb_pred_large = self.affine_output_projection(
            x_large, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )
        tensor7_affine_medium, bb_pred_medium = self.affine_output_projection(
            x_medium, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )
        tensor7_affine_small, bb_pred_small = self.affine_output_projection(
            x_small, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x_small)
        _, _, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        special_tokens_mask = structure_tokens_large >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            sequence_id=sequence_id,
            max_bin=self.max_pae_bin,
        )
        # This might be broken for chainbreak tokens? We might align to the chainbreak
        ptm = compute_tm(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            max_bin=self.max_pae_bin,
        )

        plddt_logits = self.plddt_head(x_large)
        plddt_value = CategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean()

        return dict(
            tensor7_affine_large=tensor7_affine_large,
            bb_pred_large=bb_pred_large,
            tensor7_affine_medium=tensor7_affine_medium,
            bb_pred_medium=bb_pred_medium,
            tensor7_affine_small=tensor7_affine_small,
            bb_pred_small=bb_pred_small,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
        )

    def decode_fine(
            self,
            structure_tokens_large: torch.Tensor,
            lengths: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
        ):
            B = lengths.shape[0]
            idxs = torch.arange(60, device=lengths.device).unsqueeze(0)  # shape: (1, 256)
            attention_mask = idxs < (lengths + 2).unsqueeze(1)
            structure_tokens_large = apply_special_tokens(structure_tokens_large, lengths, 4096 + 2, 4096 + 1, 4096 + 3)
            #if attention_mask is None:
            #    attention_mask = torch.ones_like(structure_tokens_large, dtype=torch.bool)

            #attention_mask = attention_mask.bool()
            if sequence_id is None:
                sequence_id = torch.zeros_like(structure_tokens_large, dtype=torch.int64)
            # not supported for now
            chain_id = torch.zeros_like(structure_tokens_large, dtype=torch.int64)

            # check that BOS and EOS are set correctly
            assert (
                structure_tokens_large[:, 0].eq(self.special_tokens["BOS"]).all()
            ), "First token in structure_tokens must be BOS token"
            assert (
                structure_tokens_large[
                    torch.arange(structure_tokens_large.shape[0]), attention_mask.sum(1) - 1
                ]
                .eq(self.special_tokens["EOS"])
                .all()
            ), "Last token in structure_tokens must be EOS token"
            assert (
                (structure_tokens_large < 0).sum() == 0
            ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

            x_large = self.embed(structure_tokens_large)
            # !!! NOTE: Attention mask is actually unused here so watch out
            x_large, _, _ = self.decoder_stack.forward(
                x_large, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
            )


            tensor7_affine_large, bb_pred_large = self.affine_output_projection(
                x_large, affine=None, affine_mask=torch.zeros_like(attention_mask)
            )


            return dict(
                tensor7_affine_large=tensor7_affine_large,
                bb_pred_large=bb_pred_large,
            )

class ProjectedCodebookModel(nn.Module):
    def __init__(self, encoder_ckpt, decoder_ckpt, proj_dim=1024, device='2'):
        device = 'cuda:' + device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        super().__init__()
        #self.encoder = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).eval()
        #trainable encoder
        self.encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).to(device)
        state_dict_encoder = torch.load(
            encoder_ckpt,
            map_location=device,
        )
        self.encoder.load_state_dict(state_dict_encoder, strict=False)
        with torch.no_grad():
            self.encoder.codebook.embeddings.requires_grad=False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.medium_codebook.parameters():
            param.requires_grad = True
        for param in self.encoder.small_codebook.parameters():
            param.requires_grad = True
        
        #freezed decoder load
        self.decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30).to(device)
        state_dict_decoder = torch.load(
            decoder_ckpt,
            map_location=device,
        )
        self.decoder.load_state_dict(state_dict_decoder,  strict=False)
        with torch.no_grad():
            self.decoder.embed.weight.copy_(state_dict_decoder['embed.weight'])
            self.decoder.embed.weight.requires_grad=False

        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder.medium_embed.parameters():
            param.requires_grad = True
        for param in self.decoder.small_embed.parameters():
            param.requires_grad = True

    def forward(self, coords, attention_mask, sequence_id, residue_index, lengths):
        #chain = ProteinChain.from_atom37(
        #    coords, sequence=None
        #)
        gt_bb_coords = coords[..., :3, :].to(torch.float32)
        structure_tokenizer = StructureTokenizer()

        left_pad = 1
        right_pad = 1

        #coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        #coords = coords.to(self.device)  # (1, L, 37, 3)
        #plddt = plddt.to(self.device)  # (1, L)
        residue_index = residue_index.to(self.device)  # (1, L)
        #_, structure_tokens, commitment_loss = self.encoder.encode(
        _,_,_, structure_tokens_large_t, structure_tokens_medium_t, structure_tokens_small_t, encoder_loss_large, encoder_loss_medium, encoder_loss_small = self.encoder.encode(
            coords, residue_index=residue_index
        )
        
        #coords = torch.squeeze(coords, dim=0)  # (L, 37, 3)  # type: ignore
        #plddt = torch.squeeze(plddt, dim=0)  # (L,)  # type: ignore
        #structure_tokens_large = torch.squeeze(structure_tokens_large, dim=0)  # (L,)  # type: ignore
        #structure_tokens_medium = torch.squeeze(structure_tokens_medium, dim=0)  # (L,)  # type: ignore
        #structure_tokens_small = torch.squeeze(structure_tokens_small, dim=0)  # (L,)  # type: ignore

        #coords = F.pad(
        #    coords, (0, 0, 0, 0, left_pad, right_pad), value=torch.inf
        #)
        #plddt = F.pad(plddt, (left_pad, right_pad), value=0)
        structure_tokens_large = apply_special_tokens(structure_tokens_large_t, lengths, 4096 + 2, 4096 + 1, 4096 + 3)
        structure_tokens_medium = apply_special_tokens(structure_tokens_medium_t, lengths, 512 + 2, 512 + 1, 512 + 3)
        structure_tokens_small = apply_special_tokens(structure_tokens_small_t, lengths, 32 + 2, 32 + 1, 32 + 3)


        # tokens = {}
        # tokens["large"] = structure_tokens_large
        # tokens["medium"] = structure_tokens_medium
        # tokens["small"] = structure_tokens_small
        
        
        # structure_tokens_large = F.pad(
        #     structure_tokens_large,
        #     (left_pad, right_pad),
        #     value=structure_tokenizer.mask_token_id,
        # )
        # structure_tokens_medium = F.pad(
        #     structure_tokens_medium,
        #     (left_pad, right_pad),
        #     value=structure_tokenizer.mask_token_id,
        # )
        # structure_tokens_small = F.pad(
        #     structure_tokens_small,
        #     (left_pad, right_pad),
        #     value=structure_tokenizer.mask_token_id,
        # )
        # structure_tokens_large[0] = structure_tokenizer.bos_token_id
        # structure_tokens_large[-1] = structure_tokenizer.eos_token_id
        # structure_tokens_medium[0] = 512 + 2
        # structure_tokens_medium[-1] = 512 + 1
        # structure_tokens_small[0] = 32 + 2
        # structure_tokens_small[-1] = 32 + 1
        # structure_tokens_large = torch.unsqueeze(structure_tokens_large, dim=0)
        # structure_tokens_medium = torch.unsqueeze(structure_tokens_medium, dim=0)
        # structure_tokens_small = torch.unsqueeze(structure_tokens_small, dim=0)
        
        #output = self.decoder.decode(structure_tokens, self.decode_proj_weight)
        output = self.decoder.decode(structure_tokens_large, structure_tokens_medium, structure_tokens_small, lengths)
        # tokens["output"] = output
        # torch.save(tokens, "checking_token_tq")
        # exit()

        # bb_coords_large: torch.Tensor = output["bb_pred_large"][
        #     0, 1:-1, ...
        # ]  # Remove BOS and EOS tokens

        # bb_coords_medium: torch.Tensor = output["bb_pred_medium"][
        #     0, 1:-1, ...
        # ]  # Remove BOS and EOS tokens

        # bb_coords_small: torch.Tensor = output["bb_pred_small"][
        #     0, 1:-1, ...
        # ]  # Remove BOS and EOS tokens
        
        bb_coords_large = output["bb_pred_large"]
        bb_coords_medium = output["bb_pred_medium"]
        bb_coords_small = output["bb_pred_small"]

        bb_dist_loss_large = backbone_distance_loss_Huber(bb_coords_large, gt_bb_coords, lengths)
        bb_dist_loss_medium = backbone_distance_loss_Huber(bb_coords_medium, gt_bb_coords, lengths)
        bb_dist_loss_small = backbone_distance_loss_Huber(bb_coords_small, gt_bb_coords, lengths)

        bb_direction_loss_large = backbone_direction_loss(bb_coords_large, gt_bb_coords, lengths)
        bb_direction_loss_medium = backbone_direction_loss(bb_coords_medium, gt_bb_coords, lengths)
        bb_direction_loss_small = backbone_direction_loss(bb_coords_small, gt_bb_coords, lengths)

        rmsd_large = compute_rmsd_aligned_torch(bb_coords_large, gt_bb_coords, lengths)
        rmsd_medium = compute_rmsd_aligned_torch(bb_coords_medium, gt_bb_coords, lengths)
        rmsd_small = compute_rmsd_aligned_torch(bb_coords_small, gt_bb_coords, lengths)

        bb_coords = [bb_coords_large, bb_coords_medium, bb_coords_small]
        bb_dist_loss = [bb_dist_loss_large, bb_dist_loss_medium, bb_dist_loss_small]
        bb_direction_loss = [bb_direction_loss_large, bb_direction_loss_medium, bb_direction_loss_small] 
        encoder_loss = [encoder_loss_large, encoder_loss_medium, encoder_loss_small]
        rmsd = [rmsd_large, rmsd_medium, rmsd_small]
        #return bb_coords, commitment_loss, bb_dist_loss, bb_direction_loss

        return bb_coords, bb_dist_loss, bb_direction_loss, encoder_loss, rmsd

def train():
    def env_int(name, default):
        value = os.getenv(name)
        return default if value in (None, "") else int(value)

    device_name = os.getenv("HCG_DEVICE", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = env_int("HCG_BATCH_SIZE", 16)
    num_workers = env_int("HCG_NUM_WORKERS", 4)
    max_epochs = env_int("HCG_MAX_EPOCHS", 200)
    max_steps = env_int("HCG_MAX_STEPS", 0)
    save_dir = os.getenv("HCG_SAVE_DIR", "/data/hier_VQ_VAE_ckpt/new_vqvae_huber")
    os.makedirs(save_dir, exist_ok=True)

    model = ProjectedCodebookModel(
        encoder_ckpt="/data/esm3_checkpoint/esm3_structure_encoder_v0.pth",
        decoder_ckpt="/data/esm3_checkpoint/esm3_structure_decoder_v0.pth"
    ).to(device)
    esm_encoder_ckpt = "/data/esm3_checkpoint/esm3_structure_encoder_v0.pth"
    esm_enc = torch.load(esm_encoder_ckpt, map_location=device)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(name)
    optimizer = torch.optim.Adam(params_to_update, lr=1e-4)



    dataset = ProteinDataset(
        txt_file="/data/pdb_data/processed_chains/dssp_success_short_256.txt",
        chain_dir="/data/pdb_data/processed_chains",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_collate_fn)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        steps_this_epoch = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=False)
        #for step, (data1, data2) in enumerate(pbar):
        for step, (atom_pos, lengths, res_ids) in enumerate(pbar):
            steps_this_epoch = step + 1
            atom_pos = atom_pos.to(device)  # (B, 256, 37, 3)
            lengths = lengths.to(device)    # (B, 256)
            res_ids = res_ids.to(device)    # (B, 256)
            

            optimizer.zero_grad()
            bb_coords, bb_dist_loss, bb_direction_loss, encoder_loss, rmsd = model(atom_pos, None, None, res_ids, lengths)
            #loss = commitment_loss + bb_dist_loss + bb_direction_loss
            loss = bb_dist_loss[1] + bb_direction_loss[1] + encoder_loss[1] + (bb_dist_loss[2] + bb_direction_loss[2]) * 0.25 + encoder_loss[2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step() 
            
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item(), "avg_loss": total_loss / (step + 1)})

            writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/distance_medium", bb_dist_loss[1].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/distance_small", bb_dist_loss[2].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/direction_medium", bb_direction_loss[1].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/direction_small", bb_direction_loss[2].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/encoder_medium", encoder_loss[1].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/encoder_small", encoder_loss[2].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/rmsd_large", rmsd[0].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/rmsd_medium", rmsd[1].item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/rmsd_small", rmsd[2].item(), epoch * len(dataloader) + step)
            #writer.add_scalar("Loss/Commitment", commitment_loss.item(), epoch * len(dataloader) + step)
            if max_steps > 0 and steps_this_epoch >= max_steps:
                break
        avg_loss = total_loss / max(steps_this_epoch, 1)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)

        
        
        if epoch % 5 == 0:
            encoder_param = model.encoder.state_dict()
            decode_param = model.decoder.state_dict()
            torch.save({
                'epoch' : epoch,
                'encoder_weight' : encoder_param,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'encoder_checkpoint_{epoch}.pth'))
            torch.save({
                'epoch' : epoch,
                'decode_proj' : decode_param,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'decoder_checkpoint_{epoch}.pth'))

    writer.close()
    
if __name__ == "__main__":
    train()
