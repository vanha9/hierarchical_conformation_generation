import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

#from models.vqvae import StructureTokenEncoder, StructureTokenDecoder
from esm.models.vqvae import *

from esm.models.codebook_data import ProteinDataset
from esm.layers.codebook_projection import ProjectedCodebook
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
import numpy as np

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

def compute_rmsd_aligned_torch(P: torch.Tensor, Q: torch.Tensor):
    """
    Kabsch 정렬 후 RMSD 계산
    """
    P_aligned, Q_cent = kabsch_align_torch(P, Q)
    diff = P_aligned - Q_cent
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
    return rmsd

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=1280):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # input과 output 차원이 다르므로 residual 연결을 위해 projection
        self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out + self.shortcut(x)  # residual connection

#loss function implement
def backbone_distance_loss(X_hat, X):
    Z_hat = X_hat.reshape(-1, 3)
    Z = X.reshape(-1, 3)


    D_pred = torch.cdist(Z_hat, Z_hat, p=2).pow(2)
    D_true = torch.cdist(Z, Z, p=2).pow(2)

    E = (D_pred - D_true).pow(2)
    E = torch.minimum(E, torch.tensor(25.0, device=E.device))
    return E.mean() 

def compute_vectors(X):
    N, CA, C = X[:, 0], X[:, 1], X[:, 2]
    C_next = torch.roll(N, shifts=-1, dims=0)

    v1 = CA - N
    v2 = C - CA
    v3 = C_next - C
    v4 = -torch.cross(v1, v2, dim=-1)
    v5 = torch.cross(v3, v1, dim=-1)
    v6 = torch.cross(v2, v3, dim=-1)

    return torch.cat([v1, v2, v3, v4, v5, v6], dim=0)

def backbone_direction_loss(X_hat, X):
    V_hat = compute_vectors(X_hat)
    #V_hat = F.normalize(compute_vectors(X_hat), dim=-1)
    V = compute_vectors(X)
    #V = F.normalize(compute_vectors(X), dim=-1)

    D_pred = V_hat @ V_hat.T
    D_true = V @ V.T

    E = (D_pred - D_true).pow(2)
    E = torch.minimum(E, torch.tensor(20.0, device=E.device))
    return E.mean()
#loss function implement

class ProjectedDecoder(StructureTokenDecoder):
    def __init__(self, *args, d_enc_codebook=128, proj_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = ResidualMLP()

    def decode(
        self,
        structure_tokens: torch.Tensor,
        #decode_proj_weight: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        encoder_codebook: EMACodebook | None=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(structure_tokens, dtype=torch.bool)

        attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        # check that BOS and EOS are set correctly
        assert (
            structure_tokens[:, 0].eq(self.special_tokens["BOS"]).all()
        ), "First token in structure_tokens must be BOS token"
        assert (
            structure_tokens[
                torch.arange(structure_tokens.shape[0]), attention_mask.sum(1) - 1
            ]
            .eq(self.special_tokens["EOS"])
            .all()
        ), "Last token in structure_tokens must be EOS token"
        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"
        structure_tokens = torch.where(structure_tokens > 1024, structure_tokens - 4096, structure_tokens)
        
        decode_embed = self.proj(encoder_codebook.embeddings)
        x = F.embedding(structure_tokens, decode_embed)
        #x = projected_embed(structure_tokens)
        # !!! NOTE: Attention mask is actually unused here so watch out
        x, _, _ = self.decoder_stack.forward(
            x, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
        )

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x)
        _, _, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
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

        plddt_logits = self.plddt_head(x)
        plddt_value = CategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean()

        return dict(
            tensor7_affine=tensor7_affine,
            bb_pred=bb_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
        )



class ProjectedCodebookModel(nn.Module):
    def __init__(self, encoder_ckpt, decoder_ckpt, proj_dim=1024, device='0'):
        device = 'cuda:' + device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        super().__init__()
        #self.encoder = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).eval()
        #trainable encoder
        self.encoder = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=1024)
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        #freezed decoder load
        self.decoder = ProjectedDecoder(d_model=1280, d_enc_codebook=128, n_heads=20, n_layers=30)
        state_dict_decoder = torch.load(
            decoder_ckpt,
            map_location=device,
        )
        self.decoder.load_state_dict(state_dict_decoder, strict=False)
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder.proj.parameters():
            param.requires_grad = True


    def forward(self, coords, attention_mask, sequence_id, residue_index):
        chain = ProteinChain.from_atom37(
            coords, sequence=None
        )
        gt_bb_coords = coords[..., :3, :]
        gt_bb_coords = torch.squeeze(gt_bb_coords, dim=0).to(self.device)
        
        #structure_tokenizer
        structure_tokenizer = StructureTokenizer()
        #structure_encoder
        #encode structure to structure tokens
        left_pad = 1
        right_pad = 1

        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.to(self.device)  # (1, L, 37, 3)
        plddt = plddt.to(self.device)  # (1, L)
        residue_index = residue_index.to(self.device)  # (1, L)
        #_, structure_tokens, commitment_loss = self.encoder.encode(
        _, structure_tokens, encoder_loss = self.encoder.encode(
            coords, residue_index=residue_index
        )
        coords = torch.squeeze(coords, dim=0)  # (L, 37, 3)  # type: ignore
        plddt = torch.squeeze(plddt, dim=0)  # (L,)  # type: ignore
        structure_tokens = torch.squeeze(structure_tokens, dim=0)  # (L,)  # type: ignore

        coords = F.pad(
            coords, (0, 0, 0, 0, left_pad, right_pad), value=torch.inf
        )
        plddt = F.pad(plddt, (left_pad, right_pad), value=0)
        structure_tokens = F.pad(
            structure_tokens,
            (left_pad, right_pad),
            value=structure_tokenizer.mask_token_id,
        )
        structure_tokens[0] = structure_tokenizer.bos_token_id
        structure_tokens[-1] = structure_tokenizer.eos_token_id
        structure_tokens = torch.unsqueeze(structure_tokens, dim=0)
        
        #output = self.decoder.decode(structure_tokens, self.decode_proj_weight)
        output = self.decoder.decode(structure_tokens, encoder_codebook=self.encoder.codebook)
        bb_coords: torch.Tensor = output["bb_pred"][
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens

        bb_dist_loss = backbone_distance_loss(bb_coords, gt_bb_coords)
        bb_direction_loss = backbone_direction_loss(bb_coords, gt_bb_coords)

        rmsd = compute_rmsd_aligned_torch(bb_coords, gt_bb_coords)

        #return bb_coords, commitment_loss, bb_dist_loss, bb_direction_loss
        return bb_coords, bb_dist_loss, bb_direction_loss, encoder_loss, rmsd

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProjectedCodebookModel(
        encoder_ckpt="/data/esm3_checkpoint/esm3_structure_encoder_v0.pth",
        decoder_ckpt="/data/esm3_checkpoint/esm3_structure_decoder_v0.pth"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    dataset = ProteinDataset(
        txt_file="/data/pdb_data/processed_chains/dssp_success_short_256.txt",
        chain_dir1="/data/pdb_data/processed_chains",
        chain_dir2="/data/pdb_data/processed_chains_second_structure"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    save_dir = "/esm/esm/models/codebook_weights"
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))

    for epoch in range(20):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=False)
        for step, (data1, data2) in enumerate(pbar):
        #for step, data in enumerate(pbar):
            # coords
            #atom_pos_aa = data1["atom_positions"].clone().detach()  # (B, L, 37, 3)
            atom_pos_ss = data2["mean_atom_positions"].clone().detach()  # (B, L, 37, 3)

            #atom_pos = torch.cat([atom_pos_ss, atom_pos_aa], dim=1)

            # Build sequence_id
            L = atom_pos_ss.shape[1]
            res_id = torch.arange(L).unsqueeze(0).to(device)
            
            #L = atom_pos_ss.shape[1]
            #res_id = torch.arange(L).unsqueeze(0).to(device)

            optimizer.zero_grad()
            #bb_coords, commitment_loss, bb_dist_loss, bb_direction_loss = model(atom_pos_ss, None, None, res_id)
            bb_coords, bb_dist_loss, bb_direction_loss, encoder_loss, rmsd = model(atom_pos_ss, None, None, res_id)
            #loss = commitment_loss + bb_dist_loss + bb_direction_loss
            loss = bb_dist_loss + bb_direction_loss + encoder_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item(), "avg_loss": total_loss / (step + 1)})

            writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/distance", bb_dist_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/direction", bb_direction_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/encoder", encoder_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/rmsd", rmsd.item(), epoch * len(dataloader) + step)
            #writer.add_scalar("Loss/Commitment", commitment_loss.item(), epoch * len(dataloader) + step)

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        
        encoder_param = model.encoder.state_dict()
        decode_proj_state = model.decoder.proj.state_dict()
        torch.save({
            'epoch' : epoch,
            'decode_proj' : decode_proj_state,
            'encoder_weight' : encoder_param,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'/esm/esm/models/codebook_weights/codebook_checkpoint_{epoch}.pth')

    writer.close()
    
if __name__ == "__main__":
    train()