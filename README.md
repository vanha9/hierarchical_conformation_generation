# Hierarchical Codebook Learning for Protein Structure Generation

This repository contains the official implementation of hierarchical discrete representations (VQ-VAE) for coarse-to-fine protein conformation generation.

### Repository Clone & Path Configuration
Clone the repository inside the container
```bash
git clone https://github.com/vanha9/hierarchical_conformation_generation.git
```

**Data Download and Preprocessing (from ESMDiff)**
Follow the procedure from the ESMDiff baseline to download and preprocess the PDB database:
```bash
# to download the whole pdb database
cd /hierarchical_conformation_generation/hierarchical_ConfGen
python scripts/download_pdb_mmcif.sh /data/pdb_data/pdb_mmcif
pip install biopython==1.79
python scripts/pdb/preprocess.py --mmcif_dir /data/pdb_data/pdb_mmcif --output_dir /data/pdb_data/processed_chains --per_chain --strip_array
pip install biopython==1.84 
python scripts/dump.py /data/pdb_data/processed_chains /data/pdb_data/processed_chains_encoding pkl
```

## Training the VQ-VAE
Once preprocessing is complete, you can train the VQ-VAE model using the main script. We have verified that the training loop runs successfully in the configured environment.
```bash
cd /hierarchical_conformation_generation/Hierarchical_VQ_VAE
export PYTHONPATH=/hierarchical_conformation_generation/Hierarchical_VQ_VAE:$PYTHONPATH
python esm/models/new_vqvae.py
```

**Data Preparation & Tokenization**
You can pre-tokenize structure using the file:
/hierarchical_conformation_generation/Hierarchical_VQ_VAE/esm/models/preprocess_to_ptm0.py

## Training the Auto-regressive Discrete diffusion model
```bash
cd /hierarchical_conformation_generation/hierarchical_ConfGen
export PYTHONPATH=/hierarchical_conformation_generation/hierarchical_ConfGen:$PYTHONPATH
python esm/models/new_vqvae.py
python train.py experiment=mdlm data.batch_size=4 logger=csv trainer.devices=4 data.train_val_split=[0.8,0.2]
```

## Sampling the structure
```bash
 python slm/sample_esmdiff_var.py --input data/targets/bpti --output /hierarchical_conformation_generation/hierarchical_ConfGen/slm/models/output/bpti --num_steps 25 --num_samples 100 --ckpt /data/hier_ConfGen_ckpt
```

[Download Model Checkpoints via Zenodo](https://zenodo.org/records/19412545)
