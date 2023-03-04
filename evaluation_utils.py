from typing import List

import torch
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from model import BaseModel
from aae import AAE
from model_utils import get_params
from dataset import MolerDataset
from rdkit import Chem


class MoLeRGenerator(DistributionMatchingGenerator):
    def __init__(self, ckpt_file_path, layer_type, model_type, using_lincs, device="cuda:0"):
        dataset = MolerDataset(
            root="/data/ongh0068",
            raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",
            output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
            split="valid_0",
        )
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
            self.model = AAE.load_from_checkpoint(
                ckpt_file_path, params=params, dataset=dataset, using_lincs = using_lincs
            )
        self.model = self.model.to(device) if device is not None else self.model.cuda()
        self.model.eval()
        

    def generate(
        self, number_samples: int, latent_space_dim: int = 512, max_num_steps: int = 120
    ) -> List[str]:
        z = torch.randn(number_samples, latent_space_dim).cuda()
        decoder_states = self.model.decode(
            latent_representations=z, max_num_steps=max_num_steps
        )
        samples = [
            Chem.MolToSmiles(decoder_state.molecule) for decoder_state in decoder_states
        ]

        return samples
