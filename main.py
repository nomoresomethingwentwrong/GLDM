import sys
import os
import argparse
import pickle
import numpy as np
import torch
from rdkit.Chem import Draw
from .latent_diffusion.moler_ldm import LatentDiffusion
from .autoencoder.dataset import DummyDataset
from omegaconf import OmegaConf
from .autoencoder.model_utils import get_params
from .latent_diffusion.DDIM import MolSampler
from rdkit.Chem.Draw import IPythonConsole
import warnings
IPythonConsole.drawOptions.addAtomIndices = False
IPythonConsole.drawOptions.addStereoAnnotation = True
IPythonConsole.drawOptions.baseFontSize = 0.5
IPythonConsole.ipython_useSVG=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, config_file):
    model_with_metadata = torch.load(model_path, map_location=device)
    checkpoint_states = model_with_metadata['model']
    metadata = model_with_metadata['meta_data']
    config = OmegaConf.load(config_file)
    ldm_params = config['model']['params']

    return checkpoint_states, metadata, config, ldm_params


def is_cond_model(config):
    return config['model']['cond_stage_config'] != '__is_unconditional__'


def instantiate_model(model_path, config_file):
    # ----------------- load model & config -----------------
    checkpoint_states, metadata, config, ldm_params = load_model(model_path, config_file)

    # ----------------- args -----------------
    batch_size = 1
    gen_step_drop_probability = 0
    dataset = DummyDataset(metadata)
    first_stage_params = get_params(dataset)
    first_stage_config = config['model']['first_stage_config']
    ldm_params = config['model']['params']
    unet_params = config['model']['unet_config']['params']
    first_stage_params["gene_exp_condition_mlp"]["input_feature_dim"] = (832 + 978 + 1) if first_stage_config['model_type'] != 'vae' else (512 + 978 + 1)
    gene_expr_dim = config['model']['cond_stage_config']['params']['dim'] if is_cond_model(config) else None

    # ----------------- instantiate model -----------------
    ldm_model = LatentDiffusion(
        first_stage_config,
        config['model']['cond_stage_config'],
        dataset, 
        gen_step_drop_probability,
        batch_size,
        first_stage_params,
        first_stage_config['ckpt_path'],
        unet_config = config['model']['unet_config'],
        **ldm_params
    )
    ldm_model.load_state_dict(checkpoint_states)
    ldm_model.to(device=device)
    ldm_model.eval()

    return ldm_model, is_cond_model(config), gene_expr_dim


def sample_from_model(model, gene_expr=None, num_samples=10):
    # ----------------- sample -----------------
    sampler = MolSampler(model)
    ddim_steps = 500
    ddim_eta = 1.0
    size = (1, 512)
    samples, _ = sampler.sample(
        S = ddim_steps,
        batch_size = num_samples,  # not batch size
        conditioning = gene_expr,
        shape = size,
        ddim_eta = ddim_eta, 
        verbose = False
    )

    decoding_samples = samples.view((num_samples, -1))
    decoding_res = model.first_stage_model.decode(decoding_samples)
    output_mols = [decoding_res[i].molecule for i in range(len(decoding_res))]

    return output_mols


def save_mols(output_mols, output_file, save_img):
    # ----------------- save -----------------
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(output_mols, f)
    if save_img is not None:
        if '.png' in save_img:
            img = Draw.MolsToGridImage(output_mols, subImgSize=(200,200), maxMols = 1000, molsPerRow=5, returnPNG=False)
            img.save(save_img)
        # elif '.svg' in save_img:
        #     img = Draw.MolsToGridImage(output_mols, subImgSize=(200,200), maxMols = 1000, molsPerRow=5, useSVG=True, returnPNG=False)
        #     with open(save_img, 'wb') as f:
        #         f.write(img.data)
            # img.save(save_img)
        else:
            raise ValueError("Invalid file format")
        

def sampleMol(model_path, config_file, gene_expression=None, gene_expression_file=None, num_samples=10, output_file='samples.pkl', save_img=None):
    do_constrained_generation = False if gene_expression is None and gene_expression_file is None else True
    if do_constrained_generation:
        if gene_expression is not None:
            if gene_expression_file is not None:
                warnings.warn("Both gene_expression and gene_expression_file were provided. Using gene_expression.")
            if isinstance(gene_expression, (np.ndarray, list)):
                gene_expr = torch.tensor(gene_expression, dtype=torch.float32).to(device)
            elif isinstance(gene_expression, torch.Tensor):
                gene_expr = gene_expression.to(device)
            else:
                raise ValueError("Invalid gene_expression type. Expected numpy array, list, or torch tensor.")
        else:
            gene_expr = torch.load(gene_expression_file, map_location=device)

    ldm_model, is_cond_model, gene_expr_dim = instantiate_model(model_path, config_file)

    if is_cond_model and do_constrained_generation:
        try:
            gene_expr = gene_expr.view(num_samples, 1, gene_expr_dim)
        except:
            raise ValueError(f"Invalid gene expression shape. Expected (num_samples, 1, {gene_expr_dim}), got {gene_expr.shape}")
    elif is_cond_model and not do_constrained_generation:
        raise ValueError("Model is a conditional model, but no gene expression was provided.")
    elif not is_cond_model and do_constrained_generation:
        warnings.warn("Model is an unconditional model, but gene expression was provided. Gene expression will be ignored.")
    else:
        gene_expr = None

    output_mols = sample_from_model(ldm_model, gene_expr, num_samples)

    save_mols(output_mols, output_file, save_img)

    return output_mols

def hello():
    print("Hello from main.py")