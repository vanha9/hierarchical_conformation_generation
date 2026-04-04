import os
from glob import glob
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from typing import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import biotite.structure as struct
from biotite.structure.io.pdb import PDBFile

from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.tokenization.tokenizer_base import EsmTokenizerBase

@dataclass
class TokenizerCollection:
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer

import esm
from esm.models.function_decoder import FunctionTokenDecoder
from esm.tokenization import (
    get_esm3_model_tokenizers,
    get_esmc_model_tokenizers,
)
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import StructureTokenizer
from esm.utils.decoding import decode_structure
from esm.utils import encoding, decoding, structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_300M,
    ESMC_600M,
)

ModelBuilder = Callable[[torch.device | str], nn.Module]


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        "/data/esm3_checkpoint/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        "/data/esm3_checkpoint/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restypes_with_x = restypes + ['X']

import pickle

path = "/data/pdb_data/processed_chains/zz/8zz0_G.pkl"
with open(path, 'rb') as f:
    data = pickle.load(f)

def decode_seq(aatype):
    return ''.join([restypes_with_x[i] for i in aatype])

sequence = decode_seq(data['aatype'])
coordinates=torch.tensor(data['atom_positions'])

print("=========before encode==========")
print(f"sequence    : {sequence}")
print(f"sequence len: {len(sequence)}")
print(f"coordinates : {coordinates.shape}")
print(coordinates[0])
print("=========before encode==========\n")

tokenizer = TokenizerCollection(
            sequence=EsmSequenceTokenizer(),
            structure=StructureTokenizer(),
            secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
            sasa=SASADiscretizingTokenizer(),
            function=InterProQuantizedTokenizer(),
            residue_annotations=ResidueAnnotationsTokenizer(),
        )

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
with torch.device(device):
    model = ESM3(
        d_model=1536,
        n_heads=24,
        v_heads=256,
        n_layers=48,
        structure_encoder_fn=ESM3_structure_encoder_v0,
        structure_decoder_fn=ESM3_structure_decoder_v0,
        function_decoder_fn=ESM3_function_decoder_v0,
        tokenizers=tokenizer,
    ).eval()

state_dict = torch.load(
    "/data/esm3_checkpoint/esm3_sm_open_v1.pth", map_location=device
)
model.load_state_dict(state_dict)

prot = ESMProtein(sequence=sequence, coordinates=coordinates)

tokens = model.encode(prot)

coords_pred, plddt, ptm = decode_structure(
    structure_tokens=tokens.structure,
    structure_decoder=model.get_structure_decoder(), 
    structure_tokenizer=model.tokenizers.structure,
    sequence=prot.sequence,
)
print("========input=========")
print(tokens.sequence.shape)
print(tokens.structure.shape)
print(tokens.coordinates.shape)
print(tokens.coordinates[0])
print(tokens.coordinates[1])
print("========input=========\n")

print("========output=========")
print(coords_pred.shape)
print("========output=========\n")

