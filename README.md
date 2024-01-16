# GLDM: Hit Molecule Generation with Constrained Graph Latent Diffusion Model

## Environment setup

Create the GLDM conda environment with the config file `GLDM.yml`
```
conda env create --file=GLDM.yml
```


## Training GLDM from scratch

To directly use trained models, please skip and refer to next section.

### Data preprocessing

We provide molecule data from GuacaMol and L1000 datasets processed by the MoLeR algorithm at this [share point](https://entuedu-my.sharepoint.com/:f:/g/personal/conghao001_e_ntu_edu_sg/EoOGZKHS5J9AnEpjnPtFbRYBgGu2Jg5K_uscjKjdBXpiFQ?e=6XewlF). If other datasets is required, please refer to [MoLeR](https://github.com/microsoft/molecule-generation) for preprocessing. 

Our gene expression data is downloaded from the [BiAAE repository](https://github.com/insilicomedicine/BiAAE). It is also available [here](https://drive.google.com/drive/folders/1cbcZZgjlV3W6D_ROVLOGX_Q6Ef-JXU3y?usp=sharing).

### Training the encoder and decoder

When excuting the training scripts for the first time, additional preprocessing will be done to convert data samples into `pytorch_geometric data`. This process will take some time. 

In addition, remember to change the following variables in the scripts:

> In *autoencoder/train_guacamol.py* and *autoencoder/train_l1000.py*, `raw_moler_trace_dataset_parent_folder` and `output_pyg_trace_dataset_parent_folder` refer to the MoLeR processed data folder and the pytorch_geometric acceptable data folder. Please unzip the trace directory downloaded from the share point, and put it under `raw_moler_trace_dataset_parent_folder`. The `pytorch_geometric data` will be automatically stored in `output_pyg_trace_dataset_parent_folder`. 

> In *autoencoder/train_l1000.py*, `gene_exp_controls_file_path` and `gene_exp_tumor_file_path` stores the gene expression profiles of control and treated cell lines, and `lincs_csv_file_path` stores the experiment idx. Please put `robust_normalized_controls.npz` under `gene_exp_controls_file_path`, put `robust_normalized_tumors.npz` under `gene_exp_tumor_file_path`, and put `experiments_filtered.csv` under `lincs_csv_file_path`.

#### Training unconstrained model on GuacaMol dataset

Train GLDM with WAE loss
```
cd autoencoder
python train_guacamol.py \
        --layer_type=FiLMConv \
        --model_architecture=aae \
        --gradient_clip_val=0.0 \
        --max_lr=1e-4 --gen_step_drop_probability=0.5 \
        --using_wasserstein_loss --using_gp
```

- To use other GNN layer type, change `--layer_type` values into `GATConv` or `GCNConv`. 
- For GLDM with other regularization losses:
    - GLDM + GAN loss: remove `--using_wasserstein_loss --using_gp`
    - GLDM + VAE loss: remove `--using_wasserstein_loss --using_gp` and change `--model_architecture=vae`

Model checkpoints will automatically be saved under the current folder. 

#### Training constrained model on L1000 dataset

```
python train_l1000.py \
    --layer_type=FiLMConv \
    --model_architecture=aae \
    --gradient_clip_val=0.0 \
    --use_oclr_scheduler \
    --max_lr=1e-4 \
    --using_wasserstein_loss --using_gp \
    --gen_step_drop_probability=0.0
```
- To try other GNN layer type or regularization loss, refer to last section to modify the parameters
- To finetune a pretrained unconstrained model, configure `--pretrained_ckpt` with the pretrained model checkpoint and `--pretrained_ckpt_model_type` to be `vae` or `aae` accordingly

### Training the latent diffusion model

Make sure the encoder model is developed before proceeding to train the latent diffusion model. Configure the path to the encoder model checkpoint in `config_file`. Also remember to change the path variables likely to training the encoder and decoder.

#### Training unconstrained model on GuacaMol dataset

```
cd ldm
python train_ldm_guacamol.py \
    --layer_type=FiLMConv \
    --model_architecture=aae \
    --use_oclr_scheduler \
    --gradient_clip_val=1.0 \
    --max_lr=1e-4 \
    --using_wasserstein_loss --using_gp \
    --gen_step_drop_probability=0.9 \
    --config_file=config/ldm_uncon+wae_uncon.yml
```

#### Training constrained model on L1000 dataset

```
python train_ldm_l1000.py \
    --layer_type=FiLMConv \
    --model_architecture=aae \
    --use_oclr_scheduler \
    --gradient_clip_val=1.0 \
    --max_lr=1e-4 \
    --using_wasserstein_loss --using_gp \
    --gen_step_drop_probability=0.9 \
    --config_file=config/ldm_con+wae_con.yml
```

## Sample hit molecules

Please download the trained models from [Zenodo](https://zenodo.org/records/10456911?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZjZmJlOWY5LTQ1MTUtNGJmZi1iZDAyLWE3NTA0OTc2M2FkMiIsImRhdGEiOnt9LCJyYW5kb20iOiIyZmI1NDhkNGY0ODIwNDFkN2E0MzIwMDFhZWFlZWE0MyJ9.Ep28OUkeVm5ksE5n0NgVSucdOpRiBnyKRuKXr4Is2-_3vS2vDI-DfbH-tczvF6EPlPmJ6Tx8qvgdAMKKkaiLJw) and refer to *ldm/sample_ldm.ipynb* for sampling from developed models.

## Evaluations

### GuacaMol benchmarks

```
python distribution_learning.py \
    --using_ldm \
    --ldm_ckpt=model_ckpt/GLDM_WAE_uncond.ckpt \
    --ldm_config=ldm/config/ldm_uncon+wae_uncon.yml \
    --output_fp=guacamol_latest_ldm_wae_10000.json \
    --smiles_file=guacamol_latest_ldm_wae_10000_smiles.pkl \
    --number_samples=10000
```
The scores of the benchmarks will be shown at the end of the output file (stored at `--output_fp`).

### Structural similarity 

First generate the hit candidates constrained by gene expression changes in the test dataset. 
```python evaluate_ldm_l1000_metrics.py -d cuda -m wae```

Then refer to *ldm/testset_rediscovery.ipynb* for evaluations. 

### Synthetic accessibility and quantitative estimate of drug-likeness

Refer to *ldm/SA_QED.ipynb* for details.

### Binding affinity 

20 protein-ligand complex structures are used as baselines. Meta information is recorded in *binding_affinity/protein/metadata.csv*. 

- Download the binding complex structures from PDB:
    ```
    cd binding_affinity
    python download_pdb.py
    ```

- Generate the molecules targeted at the selected proteins. 
    ```
    cd ldm
    python evaluate_ldm_l1000_metrics.py -d cuda -m wae -b
    ```
- Save the 3D conformers of the generated molecules 
    ```
    cd binding_affinity
    python save_ligand.py -m wae
    ```

- Molecular docking with Gnina
    ```
    python baseline_docking.py
    python molecule_docking.py -m wae
    ```
    `baseline_docking` performs docking for the original molecules and `molecule_docking` deals with the generated molecules.

Change the parameter `-m` accordingly if you want to check other models. 
Refer to *binding_affinity/evaluate_gnina.ipynb* for analyzing the results. 