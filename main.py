import sys
import os
import argparse
import pickle
from .latent_diffusion.moler_ldm import LatentDiffusion
import torch
from .autoencoder.dataset import MolerDataset
from omegaconf import OmegaConf
from .autoencoder.model_utils import get_params
from .latent_diffusion.DDIM import MolSampler

def sampleMol():
    return None

def hello():
    print("Hello from main.py")