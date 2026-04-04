import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from esm.layers.codebook import EMACodebook


class ProjectedCodebook(EMACodebook):
    def forward(self, z, W):
        # z: [b, t, c]
        if self._need_init and self.training and not self.freeze_codebook:
            self._init_embeddings(z)

        proj_embeddings = W.T @ self.embeddings
        # z is of shape [batch_size, sequence length, channels]
        flat_inputs = z.view(-1, self.embedding_dim)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ proj_embeddings.t()
            + (proj_embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )  # [bt, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])  # [b, t, ncode]

        embeddings = F.embedding(encoding_indices, proj_embeddings)  # [b, t, c]

        #commitment_loss = 0.001 * F.mse_loss(z, embeddings.detach())
        commitment_loss = 0.001 * F.mse_loss(z, embeddings)

        # EMA codebook update
        #if self.training and not self.freeze_codebook:
        #    assert False, "Not implemented"
        #embeddings_st = (embeddings - z).detach() + z
        embeddings_st = (embeddings - z) + z

        return embeddings_st, encoding_indices, commitment_loss
