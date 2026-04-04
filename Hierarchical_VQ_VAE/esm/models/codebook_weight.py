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

class ProjectedEncoder(StructureTokenEncoder):
    def __init__(self, *args, proj_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_proj = ProjectedCodebook

    
    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, _ = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )
            B, L, E = knn_edges.shape

            affine_tensor = affine.tensor  # for easier manipulation
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            )
            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            )
            knn_chain_id = torch.zeros(
                B * L, E, dtype=torch.int64, device=coords.device
            )

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                )

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)

        z, _, _ = self.transformer.forward(
            x=z,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        )

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]

        return z

    @staticmethod
    def find_knn_edges(
        coords,
        padding_mask,
        coord_mask,
        sequence_id: torch.Tensor | None = None,
        knn: int | None = None,
    ) -> tuple:
        assert knn is not None, "Must specify a non-null knn to find_knn_edges"
        # Coords are N, CA, C
        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca, coord_mask, padding_mask, sequence_id, no_knn=knn
            )

        return edges, edge_mask

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        encode_proj_weight: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()

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

        z_q, min_encoding_indices, commitment_loss = self.codebook_proj(z, encode_proj_weight)

        return z_q, min_encoding_indices, commitment_loss

class ProjectedDecoder(StructureTokenDecoder):
    def __init__(self, *args, proj_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_embed = nn.Embedding(
            proj_dim + len(self.special_tokens), kwargs['d_model']
        )

    def decode(
        self,
        structure_tokens: torch.Tensor,
        #decode_proj_weight: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
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
        x = self.decode_embed(structure_tokens)
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
        #freezed encoder load
        self.encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096, proj_dim=proj_dim)
        state_dict_encoder = torch.load(
            encoder_ckpt,
            map_location=device,
        )
        self.encoder.load_state_dict(state_dict_encoder, strict=False)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        #freezed decoder load
        self.decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
        state_dict_decoder = torch.load(
            decoder_ckpt,
            map_location=device,
        )
        self.decoder.load_state_dict(state_dict_decoder,  strict=False)
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder.decode_embed.parameters():
            param.requires_grad = True

        #codebook function override
        original_codebook = self.encoder.codebook  # keep reference to original EMA codebook
        self.codebook = ProjectedCodebook(
            n_codes=original_codebook.embeddings.shape[0],
            embedding_dim=original_codebook.embeddings.shape[1],
        )
        self.codebook.load_state_dict(original_codebook.state_dict(), strict=True)
        self.encoder.codebook_proj = self.codebook
        
        # Projection weight
        self.encode_proj_weight = nn.Parameter(torch.randn(4096, proj_dim))  # W
        self.encode_proj_weight.requires_grad = True

        #self.decode_proj_weight = nn.Parameter(torch.randn(4096, proj_dim))
        #self.decode_proj_weight.requires_grad = True


    def forward(self, coords, attention_mask, sequence_id, residue_index):
        chain = ProteinChain.from_atom37(
            coords, sequence=None
        )
        gt_bb_coords = coords[..., :3, :]
        #gt_bb_coords = coords[..., :3, :]
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
        _, structure_tokens, commitment_loss = self.encoder.encode(
            coords, residue_index=residue_index, encode_proj_weight=self.encode_proj_weight
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
        output = self.decoder.decode(structure_tokens)
        bb_coords: torch.Tensor = output["bb_pred"][
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens

        bb_dist_loss = backbone_distance_loss(bb_coords, gt_bb_coords)
        bb_direction_loss = backbone_direction_loss(bb_coords, gt_bb_coords)

        return bb_coords, commitment_loss, bb_dist_loss, bb_direction_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProjectedCodebookModel(
        encoder_ckpt="/data/esm3_checkpoint/esm3_structure_encoder_v0.pth",
        decoder_ckpt="/data/esm3_checkpoint/esm3_structure_decoder_v0.pth"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = ProteinDataset(
        txt_file="/data/pdb_data/processed_chains/dssp_success_short_256.txt",
        chain_dir1="/data/pdb_data/processed_chains",
        chain_dir2="/data/pdb_data/processed_chains_second_structure"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    save_dir = "/esm/esm/models/codebook_weights"
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))

    for epoch in range(5):
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
            bb_coords, commitment_loss, bb_dist_loss, bb_direction_loss = model(atom_pos_ss, None, None, res_id)
            #loss = commitment_loss + bb_dist_loss + bb_direction_loss
            loss = bb_dist_loss + bb_direction_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item(), "avg_loss": total_loss / (step + 1)})

            writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/distance", bb_dist_loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar("Loss/direction", bb_direction_loss.item(), epoch * len(dataloader) + step)
            #writer.add_scalar("Loss/Commitment", commitment_loss.item(), epoch * len(dataloader) + step)

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        
        encode_proj_weight = model.encode_proj_weight.detach().cpu()
        decode_embed_state = self.decoder.decode_embed.state_dict()
        torch.save({
            'epoch' : epoch,
            'decode_embed' : decode_embed_state.detach().cpu(),
            'encode_weight' : encode_proj_weight,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'/esm/esm/models/codebook_weights/codebook_checkpoint_{epoch}.pth')

    writer.close()
    
if __name__ == "__main__":
    train()