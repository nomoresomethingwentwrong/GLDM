from typing import List
import pickle
import torch
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from model import BaseModel
from aae import AAE
from ldm.moler_ldm import LatentDiffusion
from ldm.DDIM import MolSampler
from omegaconf import OmegaConf
from model_utils import get_params
from dataset import MolerDataset
from rdkit import Chem


class MoLeRGenerator(DistributionMatchingGenerator):
    def __init__(
        self, 
        ckpt_file_path, 
        layer_type, 
        model_type, 
        using_lincs, 
        using_wasserstein_loss,
        using_gp,
        device="cuda:0",
    ):
        dataset = MolerDataset(
            root="/data/ongh0068",
            raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",
            output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
            split="valid_0",
        )
        self._device = device
        params = get_params(dataset)
        ###################################################
        params['full_graph_encoder']['layer_type'] = layer_type
        params['partial_graph_encoder']['layer_type'] = layer_type
        # params['using_cyclical_anneal'] = True
        ###################################################
        if model_type == 'vae':
            self.model = BaseModel.load_from_checkpoint(
                ckpt_file_path, params=params, dataset=dataset, using_lincs = using_lincs
            )
        elif model_type == 'aae':
            if using_lincs:
                params["gene_exp_condition_mlp"]["input_feature_dim"] = 832 + 978 + 1
            self.model = AAE.load_from_checkpoint(
                ckpt_file_path, 
                params=params, 
                dataset=dataset, 
                using_lincs = using_lincs,
                using_wasserstein_loss = using_wasserstein_loss,
                using_gp = using_gp
            )
        self.model = self.model.to(self._device) if self._device is not None else self.model.cuda()
        self.model.eval()
        

    def generate(
        self, number_samples: int, latent_space_dim: int = 512, max_num_steps: int = 120
    ) -> List[str]:
        z = torch.randn(number_samples, latent_space_dim).to(self._device) if self._device is not None else torch.randn(number_samples, latent_space_dim).cuda()
        decoder_states = self.model.decode(
            latent_representations=z, max_num_steps=max_num_steps
        )
        samples = [
            Chem.MolToSmiles(decoder_state.molecule) for decoder_state in decoder_states
        ]

        return samples


class LDMGenerator(DistributionMatchingGenerator):
    def __init__(
        self, 
        ldm_ckpt, 
        ldm_config,
        number_samples = 2000,
        ddim_steps = 50,
        ddim_eta = 1.0,
        device="cuda:0",
        smiles_file = None,
    ):
        # super().__init__(
        #     ckpt_file_path, 
        #     layer_type, 
        #     model_type, 
        #     using_lincs, 
        #     using_wasserstein_loss,
        #     using_gp,
        #     device
        # )
        ckpt = torch.load(ldm_ckpt, map_location = device)
        config = OmegaConf.load(ldm_config)
        ldm_params = config['model']['params']

        dataset = MolerDataset(
            root="/data/ongh0068",
            raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",
            output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
            split="valid_0",
        )

        first_stage_params = get_params(dataset)
        first_stage_config = config['model']['first_stage_config']
        ldm_params = config['model']['params']
        unet_params = config['model']['unet_config']['params']
        batch_size = 1
        drop_prob = 0.0
        latent_space_dim = int(ldm_params['image_size'])
        size = [1, latent_space_dim]

        ldm_model = LatentDiffusion(
            first_stage_config,
            config['model']['cond_stage_config'],
            dataset, 
            drop_prob,
            batch_size,
            first_stage_params,
            first_stage_config['ckpt_path'],
            unet_config = config['model']['unet_config'],
            **ldm_params
        )
        ldm_model.load_state_dict(ckpt['state_dict'])
        ldm_model = ldm_model.to(device)
        self.model = ldm_model

        sampler = MolSampler(ldm_model)

        # number_samples = 10000   # this is used by Guacamol benchmark
        size = [1, latent_space_dim]
        z_samples, _ = sampler.sample(
            S = ddim_steps,
            batch_size = number_samples,  # not batch size
            shape = size,
            ddim_eta = ddim_eta
        )
        self.z = z_samples.view((number_samples, latent_space_dim))

        self.smiles_file = smiles_file

    def instantiate_ldm(self):
        return 
    
    def instantiate_sampler(self):
        return
    
    def generate(
        self, number_samples: int, latent_space_dim: int = 512, max_num_steps: int = 120, ddim_steps: int = 50, ddim_eta: float = 1.0
    ) -> List[str]:
        # z = torch.randn(number_samples, latent_space_dim).to(self._device) if self._device is not None else torch.randn(number_samples, latent_space_dim).cuda()
        
        # size = [1, latent_space_dim]
        # z_samples, _ = self.sampler.sample(
        #     S = ddim_steps,
        #     batch_size = number_samples,  # not batch size
        #     shape = size,
        #     ddim_eta = ddim_eta
        # )
        # z = z_samples.view((number_samples, latent_space_dim))

        decoder_states = self.model.first_stage_model.decode(
            latent_representations=self.z, max_num_steps=max_num_steps
        )
        samples = [
            Chem.MolToSmiles(decoder_state.molecule) for decoder_state in decoder_states
        ]

        if self.smiles_file is not None:
            with open(self.smiles_file, 'wb') as f:
                pickle.dump(samples, f)

        return samples