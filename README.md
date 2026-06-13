# Hierarchical Discrete Representations for Protein Conformation Generation

This repository contains the implementation of **Hierarchical Discrete Representations for Coarse-to-Fine Protein Conformation Generation**.

The code has two main uses:

1. Generate protein conformational ensembles with the released model.
2. Reproduce or retrain the VQ-VAE and discrete diffusion models.

If you only want to try the model on an example protein, start with **Quick Start: Inference**.

## Availability

Code repository:

```text
https://github.com/vanha9/hierarchical_conformation_generation
```

Released checkpoints are archived on Zenodo:

```text
https://zenodo.org/records/19412545
DOI: 10.5281/zenodo.19412545
```

The Zenodo record contains:

```text
encoder_checkpoint_110.pth
decoder_checkpoint_110.pth
mp_rank_00_model_states.pt
```

For inference with the released diffusion model, use:

```text
mp_rank_00_model_states.pt
```

The encoder and decoder checkpoints are used for the trained hierarchical VQ-VAE components.

## Repository Layout

```text
Hierarchical_VQ_VAE/
  VQ-VAE model, codebook model, and structure-token preprocessing code

hierarchical_ConfGen/
  discrete diffusion model, sampling script, training configuration, and example targets

hierarchical_ConfGen/data/targets/bpti/
  small example input used in the quick start

environment.yml
  conda environment used for the experiments
```

## Requirements

The commands below assume:

- Linux
- NVIDIA GPU
- Docker with NVIDIA GPU support
- Conda inside the Docker image

The environment was tested with:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

Other CUDA/PyTorch environments may also work, but the commands below use this image to keep the setup reproducible.

## Common Setup

### 1. Clone the repository

```bash
git clone https://github.com/vanha9/hierarchical_conformation_generation.git
cd hierarchical_conformation_generation
```

### 2. Start a Docker container

The examples below mount the repository to `/hierarchical_conformation_generation` and a data directory to `/data`.

```bash
docker run --gpus '"device=0,1,2,3"' --ipc=host --shm-size=32g -it \
  --name hcg_dev \
  --mount src=$(pwd),dst=/hierarchical_conformation_generation,type=bind \
  --mount src=/path/to/your/data,dst=/data,type=bind \
  -w /hierarchical_conformation_generation \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash
```

If you want to use all visible GPUs, you can replace the first line with:

```bash
docker run --gpus all --ipc=host --shm-size=32g -it \
```

When you return to the container later:

```bash
docker start hcg_dev
docker exec -it hcg_dev bash
```

### 3. Install basic command-line tools

The tested PyTorch Docker image does not include `git`, `wget`, or `curl` by default. Install them inside the container:

```bash
apt-get update
apt-get install -y git wget curl
```

`wget` is used below to download the released checkpoints.

### 4. Create the conda environment

Inside the container:

```bash
cd /hierarchical_conformation_generation
conda env create -f environment.yml
source /opt/conda/etc/profile.d/conda.sh
conda activate hier_confgen
```

Check that PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

You should see `cuda: True`.

## Checkpoint Download

Create a directory for the released checkpoints:

```bash
mkdir -p /data/hier_ConfGen_ckpt
cd /data/hier_ConfGen_ckpt
```

Download the diffusion checkpoint used for inference. This file is large, so `-c` lets `wget` resume the download if the connection is interrupted.

```bash
wget -c -O mp_rank_00_model_states.pt \
  https://zenodo.org/api/records/19412545/files/mp_rank_00_model_states.pt/content
```

Download the VQ-VAE checkpoints if you plan to inspect or reuse the trained encoder/decoder:

```bash
wget -c -O encoder_checkpoint_110.pth \
  https://zenodo.org/api/records/19412545/files/encoder_checkpoint_110.pth/content

wget -c -O decoder_checkpoint_110.pth \
  https://zenodo.org/api/records/19412545/files/decoder_checkpoint_110.pth/content
```

The expected layout is:

```text
/data/hier_ConfGen_ckpt/
  mp_rank_00_model_states.pt
  encoder_checkpoint_110.pth
  decoder_checkpoint_110.pth
```

Optional integrity check:

```bash
stat -c '%n %s' /data/hier_ConfGen_ckpt/*
md5sum /data/hier_ConfGen_ckpt/*
```

Expected byte sizes:

```text
decoder_checkpoint_110.pth 2483418455
encoder_checkpoint_110.pth 129087994
mp_rank_00_model_states.pt 5634910416
```

Expected MD5 checksums:

```text
0ac4f398181534b31aeff59e7cb7ea94  decoder_checkpoint_110.pth
c100495eb3b13b41318e1c365bd6be58  encoder_checkpoint_110.pth
f1f5d3e255283e0f89e2fbf47314e633  mp_rank_00_model_states.pt
```

## Quick Start: Inference

This example generates conformations for BPTI using the released checkpoint.

The input is already included:

```text
hierarchical_ConfGen/data/targets/bpti/bpti.pdb
```

Run a small test first:

```bash
cd /hierarchical_conformation_generation/hierarchical_ConfGen
export PYTHONPATH=/hierarchical_conformation_generation/hierarchical_ConfGen:$PYTHONPATH

python slm/sample_esmdiff_var.py \
  --input data/targets/bpti \
  --output outputs/bpti_demo \
  --num_steps 5 \
  --num_samples 2 \
  --ckpt /data/hier_ConfGen_ckpt/mp_rank_00_model_states.pt
```

On a shared GPU server, first check which GPU has free memory:

```bash
nvidia-smi
```

Then restrict the run to one available GPU if needed:

```bash
CUDA_VISIBLE_DEVICES=2 python slm/sample_esmdiff_var.py \
  --input data/targets/bpti \
  --output outputs/bpti_demo \
  --num_steps 5 \
  --num_samples 2 \
  --ckpt /data/hier_ConfGen_ckpt/mp_rank_00_model_states.pt
```

The first inference run may also download the ESM3 model files from Hugging Face. Later runs should reuse the local cache.

This writes a generated ensemble under:

```text
hierarchical_ConfGen/outputs/bpti_demo/
```

The script creates a timestamped subdirectory. Inside it, you should find a PDB file such as:

```text
bpti.pdb
```

The output PDB is a multi-model ensemble. Each `MODEL` block corresponds to one sampled conformation.

For a larger run, increase the number of samples and denoising steps:

```bash
python slm/sample_esmdiff_var.py \
  --input data/targets/bpti \
  --output outputs/bpti \
  --num_steps 25 \
  --num_samples 100 \
  --ckpt /data/hier_ConfGen_ckpt/mp_rank_00_model_states.pt
```

### Inference Arguments

```text
--input
  Directory containing one or more .pdb files.

--output
  Directory where generated structures will be written.

--num_steps
  Number of diffusion denoising steps.

--num_samples
  Number of conformations to generate for each input PDB.

--ckpt
  Path to the released diffusion checkpoint.
```

## Notebook Tutorial

A notebook version of the BPTI example is available at:

```text
examples/quickstart_bpti_prediction.ipynb
```

It checks the environment, runs the same inference command on `data/targets/bpti`, finds the generated PDB file, and visualizes the input and output structures.

Before running all cells, check the first code cell and update `CHECKPOINT_PATH`, `OUTPUT_DIR`, or `CUDA_VISIBLE_DEVICES` if your local paths or GPU assignment are different.

For notebook visualization, install `py3Dmol` if it is not already available:

```bash
pip install py3Dmol
```

Then start Jupyter from the repository root:

```bash
cd /hierarchical_conformation_generation
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

Open:

```text
examples/quickstart_bpti_prediction.ipynb
```

## Training and Reproduction

The sections below are for retraining or reproducing the pipeline. They are not required for running inference with the released checkpoint.

## Data Download and Preprocessing

The preprocessing follows the ESMDiff data pipeline.

```bash
cd /hierarchical_conformation_generation/hierarchical_ConfGen
bash scripts/download_pdb_mmcif.sh /data/pdb_data/pdb_mmcif

python scripts/pdb/preprocess.py \
  --mmcif_dir /data/pdb_data/pdb_mmcif \
  --output_dir /data/pdb_data/processed_chains \
  --per_chain \
  --strip_array

python scripts/dump.py \
  /data/pdb_data/processed_chains \
  /data/pdb_data/processed_chains_encoding \
  pkl
```

The original ESMDiff preprocessing workflow used Biopython 1.79 for part of preprocessing. The current environment uses Biopython 1.84 because the model code imports `Bio.Data.PDBData`. If you are reproducing the legacy preprocessing step exactly and run into parser differences, use a temporary preprocessing environment with Biopython 1.79, then return to the main `hier_confgen` environment for training and inference.

Expected processed data paths:

```text
/data/pdb_data/processed_chains
/data/pdb_data/processed_chains_encoding
/data/pdb_data/processed_chains/dssp_success_short_256.txt
```

If these directories already exist on your system, you do not need to download and preprocess the full PDB database again.

## Train the Hierarchical VQ-VAE

```bash
cd /hierarchical_conformation_generation/Hierarchical_VQ_VAE
export PYTHONPATH=/hierarchical_conformation_generation/Hierarchical_VQ_VAE:$PYTHONPATH
python esm/models/new_vqvae.py
```

For a short smoke test of the training loop and checkpoint saving:

```bash
HCG_MAX_EPOCHS=1 HCG_MAX_STEPS=5 python esm/models/new_vqvae.py
```

The full script writes checkpoints to:

```text
/data/hier_VQ_VAE_ckpt/new_vqvae_huber
```

## Structure Tokenization

Structure pre-tokenization is handled by:

```text
/hierarchical_conformation_generation/Hierarchical_VQ_VAE/esm/models/preprocess_to_ptm0.py
```

Use this step when preparing tokenized structure data for diffusion model training.

## Train the Discrete Diffusion Model

The default diffusion data configuration expects:

```text
/data/pdb_data/processed_chains/dssp_success_short_256.txt
/data/structure_preprocess
/data/pdb_data/processed_chains_encoding
```

Run a short one-step smoke test first:

```bash
cd /hierarchical_conformation_generation/hierarchical_ConfGen
export PYTHONPATH=/hierarchical_conformation_generation/hierarchical_ConfGen:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python slm/train.py \
  experiment=mdlm \
  trainer=gpu \
  callbacks=none \
  logger=csv \
  data.batch_size=1 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.precision=32 \
  +trainer.max_steps=1 \
  +trainer.limit_val_batches=0 \
  trainer.default_root_dir=outputs/diffusion_smoke \
  hydra.run.dir=outputs/diffusion_smoke/hydra
```

On a shared server, replace `CUDA_VISIBLE_DEVICES=0` with an idle GPU. The `trainer.precision=32` override is intentional for this single-GPU smoke test.

To check the multi-GPU DeepSpeed path, run a two-GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python slm/train.py \
  experiment=mdlm \
  trainer=deepspeed \
  callbacks=none \
  logger=csv \
  data.batch_size=2 \
  data.num_workers=0 \
  trainer.devices=2 \
  trainer.max_epochs=1 \
  trainer.precision=32 \
  +trainer.max_steps=1 \
  +trainer.limit_val_batches=0 \
  trainer.default_root_dir=outputs/deepspeed_smoke \
  hydra.run.dir=outputs/deepspeed_smoke/hydra
```

On a shared server, choose idle GPUs in `CUDA_VISIBLE_DEVICES`.

For full training:

```bash
cd /hierarchical_conformation_generation/hierarchical_ConfGen
export PYTHONPATH=/hierarchical_conformation_generation/hierarchical_ConfGen:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python slm/train.py \
  experiment=mdlm \
  trainer=deepspeed \
  data.batch_size=4 \
  logger=csv \
  trainer.devices=4 \
  data.train_val_split=[0.8,0.2]
```

Training uses Hydra configuration files under:

```text
hierarchical_ConfGen/configs/
```

The default PDB data configuration is:

```text
hierarchical_ConfGen/configs/data/pdb.yaml
```

## Troubleshooting

### `CondaError: Run 'conda init' before 'conda activate'`

Inside the Docker image, run:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hier_confgen
```

### `ImportError: cannot import name 'PDBData' from 'Bio.Data'`

Check the Biopython version:

```bash
python - <<'PY'
import Bio
print(Bio.__version__)
from Bio.Data import PDBData
print("PDBData import ok")
PY
```

If this fails, update the environment from the repository root:

```bash
conda env update -n hier_confgen -f environment.yml
```

### `ModuleNotFoundError: No module named 'slm'` or `No module named 'esm'`

Set `PYTHONPATH` for the part of the code you are running.

For inference and diffusion training:

```bash
export PYTHONPATH=/hierarchical_conformation_generation/hierarchical_ConfGen:$PYTHONPATH
```

For VQ-VAE training:

```bash
export PYTHONPATH=/hierarchical_conformation_generation/Hierarchical_VQ_VAE:$PYTHONPATH
```

### Checkpoint not found

Check that this file exists:

```bash
ls -lh /data/hier_ConfGen_ckpt/mp_rank_00_model_states.pt
```

If it does not exist, download it from Zenodo as described in **Checkpoint Download**.

### CUDA out of memory

On a shared server, another process may already be using the default GPU. Check GPU memory first:

```bash
nvidia-smi
```

Then choose an available GPU:

```bash
CUDA_VISIBLE_DEVICES=2 python slm/sample_esmdiff_var.py ...
```

For inference, reduce:

```text
--num_samples
--num_steps
```

For training, reduce the batch size or number of devices in the Hydra command.

## Citation

If you use this repository or the released checkpoints, please cite the manuscript and the Zenodo checkpoint archive:

```text
Checkpoint archive: https://doi.org/10.5281/zenodo.19412545
```
