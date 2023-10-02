import os
import sys
sys.path.append('../')
from dataset import LincsDataset
# from model import BaseModel
# from aae import AAE
from model_utils import get_params
from rdkit.Chem import RDConfig
import itertools
from rdkit import Chem

import pandas as pd
from rdkit import RDLogger
import pickle
from l1000_evaluation_utils import compute_max_similarity
from ldm.moler_ldm import LatentDiffusion
from ldm.DDIM import MolSampler
from omegaconf import OmegaConf
import argparse

lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
import torch
import numpy as np


def generate_similar_molecules_with_gene_exp_diff(
    control_idx,
    tumour_idx,
    original_idx,
    dataset,
    model,
    device,
    rand_vect_dim=512,
    num_samples=20,
    ddim_steps=500,
    ddim_eta=1.0,
):
    # print('device: ', device)
    # model.to(device=device)
    sampler = MolSampler(model)
    # sampler = sampler.to(device=device)
    possible_pairs = np.array(list(itertools.product(control_idx, tumour_idx)))

    control_idx_batched = possible_pairs[:, 0]
    tumour_idx_batched = possible_pairs[:, 1]

    control_gene_exp_batched = dataset._gene_exp_controls[control_idx_batched]
    tumour_gene_exp_batched = dataset._gene_exp_tumour[tumour_idx_batched]
    difference_gene_exp_batched = tumour_gene_exp_batched - control_gene_exp_batched

    # Create num_samples//num_diff_vectors random vectors
    if num_samples > difference_gene_exp_batched.shape[0]:
        num_rand_vectors_required = num_samples // difference_gene_exp_batched.shape[0]
        random_vectors = torch.randn(
            num_rand_vectors_required, rand_vect_dim, device=device
        )
        # repeat each gene expression difference vector in its place a number of times
        # equal to the number of random vectors using repeat_interleave
        # then repeat the random vectors batchwise so that we can align the random vectors
        # with the gene expression differences
        # Eg given 114 gene expression diff vectors, we will have 8 random vectors
        # then for each gene expresison vector, we want to match it with each of the
        # 8 random vectors individually
        difference_gene_exp_batched = torch.tensor(
            difference_gene_exp_batched, device=device
        )
        difference_gene_exp_batched = torch.repeat_interleave(
            difference_gene_exp_batched, num_rand_vectors_required, dim=0
        )
        random_vectors = random_vectors.repeat(possible_pairs.shape[0], 1)

    else:
        num_rand_vectors_required = num_samples
        # since number of samples is less than the number of gene expressions
        # we need to truncate the gene expressions too
        difference_gene_exp_batched = torch.tensor(
            difference_gene_exp_batched[:num_samples, :], device=device
        )
        random_vectors = torch.randn(
            num_rand_vectors_required, rand_vect_dim, device=device
        )

    dose_batched = (
        torch.from_numpy(
            np.repeat(
                dataset._experiment_idx_to_dose[original_idx], (random_vectors.shape[0])
            )
        )
        .float()
        .to(device=device)
    )

    # print("dose batched shape: ", dose_batched.shape)
    # print("difference gene exp batched shape: ", difference_gene_exp_batched.shape)

    cond_vec = torch.cat((difference_gene_exp_batched, dose_batched.unsqueeze(-1)), dim=1)
    conditioning = cond_vec.view((num_samples, 1, cond_vec.size(-1)))
    # print(conditioning.device)

    size = [1, rand_vect_dim]
    conditioned_random_vectors, _ = sampler.sample(
        S = ddim_steps,
        batch_size = num_samples,  # not batch size
        conditioning = conditioning,
        shape = size,
        ddim_eta = ddim_eta,
        verbose=False
    )
    # print("cond samples device: ", conditioned_random_vectors.device)
    conditioned_random_vectors = conditioned_random_vectors.view((num_samples, rand_vect_dim))

    # compute similarity score between all 1000 generated molecules and the actual molecule
    # take the max similarity score
    decoder_states = model.first_stage_model.decode(
        latent_representations=conditioned_random_vectors, max_num_steps=120
    )
    molecules = [decoder_state.molecule for decoder_state in decoder_states]

    return molecules


def create_tensors_gene_exp_diff(
    control_idx,
    tumour_idx,
    original_idx,
    dataset,
    num_samples=20,
):
    possible_pairs = np.array(list(itertools.product(control_idx, tumour_idx)))

    control_idx_batched = possible_pairs[:, 0]
    tumour_idx_batched = possible_pairs[:, 1]

    control_gene_exp_batched = dataset._gene_exp_controls[control_idx_batched]
    tumour_gene_exp_batched = dataset._gene_exp_tumour[tumour_idx_batched]
    difference_gene_exp_batched = tumour_gene_exp_batched - control_gene_exp_batched

    # Create num_samples//num_diff_vectors random vectors
    if num_samples > difference_gene_exp_batched.shape[0]:
        num_rand_vectors_required = num_samples // difference_gene_exp_batched.shape[0]
        random_vectors = torch.randn(num_rand_vectors_required, 512)
        # repeat each gene expression difference vector in its place a number of times
        # equal to the number of random vectors using repeat_interleave
        # then repeat the random vectors batchwise so that we can align the random vectors
        # with the gene expression differences
        # Eg given 114 gene expression diff vectors, we will have 8 random vectors
        # then for each gene expresison vector, we want to match it with each of the
        # 8 random vectors individually
        difference_gene_exp_batched = torch.tensor(difference_gene_exp_batched)
        difference_gene_exp_batched = torch.repeat_interleave(
            difference_gene_exp_batched, num_rand_vectors_required, dim=0
        )
        random_vectors = random_vectors.repeat(possible_pairs.shape[0], 1)

    else:
        num_rand_vectors_required = num_samples
        # since number of samples is less than the number of gene expressions
        # we need to truncate the gene expressions too
        difference_gene_exp_batched = torch.tensor(
            difference_gene_exp_batched[:num_samples, :]
        )
        random_vectors = torch.randn(num_rand_vectors_required, 512)

    dose_batched = torch.from_numpy(
        np.repeat(
            dataset._experiment_idx_to_dose[original_idx], (random_vectors.shape[0])
        )
    ).float()

    return random_vectors, difference_gene_exp_batched, dose_batched, original_idx


class GeneExpDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        list_of_random_vectors,
        list_of_difference_gene_exp_batched,
        list_of_dose_batched,
        list_of_original_idxes,
    ):
        self.list_of_random_vectors = list_of_random_vectors
        self.list_of_difference_gene_exp_batched = list_of_difference_gene_exp_batched
        self.list_of_dose_batched = list_of_dose_batched
        self.list_of_original_idxes = list_of_original_idxes

    def __len__(self):
        return len(self.original_idxes)

    def __get_item__(self, idx):
        return (
            self.list_of_random_vectors[idx],
            self.list_of_difference_gene_exp_batched[idx],
            self.list_of_dose_batched[idx],
            self.list_of_original_idxes[idx],
        )


def sanitise(row):
    """Specifically for the L1000 csv"""
    control_indices = (
        row["ControlIndices"]
        .replace("[", "")
        .replace("]", "")
        .replace("\n", "")
        .split(" ")
    )
    control_indices = [idx for idx in control_indices if idx != ""]
    row["ControlIndices"] = np.asarray(control_indices, dtype=np.int32)
    tumour_indices = (
        row["TumourIndices"]
        .replace("[", "")
        .replace("]", "")
        .replace("\n", "")
        .split(" ")
    )
    tumour_indices = [idx for idx in tumour_indices if idx != ""]
    row["TumourIndices"] = np.asarray(tumour_indices, dtype=np.int32)
    return row


dataset = LincsDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/already_batched",
    split="valid_0",
    gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
    gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
    lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
)

# test_set = pd.read_csv("/data/ongh0068/l1000/l1000_biaae/INPUT_DIR/test.csv")
test_set = pd.read_csv("filtered_test_set.csv")
test_set = test_set.apply(lambda x: sanitise(x), axis=1)


reference_smiles = test_set.SMILES.to_list()
control_idxes = test_set.ControlIndices.values
tumour_idxes = test_set.TumourIndices.values
original_idxes = test_set.original_idx.to_list()

# Run this script with the following command:
# Add CUDA_VISIBLE_DEVICES explicitly to avoid creating tensors on disjunct GPUs
# CUDA_VISIBLE_DEVICES=0 python evaluate_ldm_l1000_metrics.py -d cuda -m vae
# CUDA_VISIBLE_DEVICES=0 python evaluate_ldm_l1000_metrics.py -d cuda -m aae
# CUDA_VISIBLE_DEVICES=0 python evaluate_ldm_l1000_metrics.py -d cuda -m wae

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_type", type=str, choices=["vae", "aae", "wae", "test"])
parser.add_argument("-d", "--device", type=str, default="cuda:0")
args = parser.parse_args()
# if args.model_type == "vae":
#     config_file = "config/ldm_con+vae_uncon.yml"
#     # ckpt_file = "tmp_logs/2023-05-08_13_39_31.158242/epoch=47-val_loss=0.12.ckpt"
#     ckpt_file = "lightning_logs/2023-05-08_13_39_31.158242/epoch=78-val_loss=0.10.ckpt"
#     output_file = "cond_generation_res/ldm_uncon_vae_generated_molecules_and_sa_scores.pkl"
#     mol_file = "cond_generation_res/ldm_uncon_vae_test_set_smile_to_max_sim_generated_molecule.pkl"
# elif args.model_type == "aae":
#     config_file = "config/ldm_con+aae_uncon.yml"
#     # ckpt_file = "tmp_logs/2023-05-08_13_39_25.455320/epoch=47-val_loss=0.11.ckpt"
#     ckpt_file = "lightning_logs/2023-05-08_13_39_25.455320/epoch=96-val_loss=0.09.ckpt"
#     output_file = "cond_generation_res/ldm_uncon_aae_generated_molecules_and_sa_scores.pkl"
#     mol_file = "cond_generation_res/ldm_uncon_aae_test_set_smile_to_max_sim_generated_molecule.pkl"
# elif args.model_type == "wae":
#     config_file = "config/ldm_con+wae_uncon.yml"
#     # ckpt_file = "tmp_logs/2023-05-08_13_39_19.133896/epoch=46-val_loss=0.13.ckpt"
#     ckpt_file = "lightning_logs/2023-05-08_13_39_19.133896/epoch=84-val_loss=0.11.ckpt"
#     output_file = "cond_generation_res/ldm_uncon_wae_generated_molecules_and_sa_scores.pkl"
#     mol_file = "cond_generation_res/ldm_uncon_wae_test_set_smile_to_max_sim_generated_molecule.pkl"
# else:
#     raise ValueError("model type not supported")

if args.model_type == "vae":
    config_file = "config/ldm_con+vae_con.yml"
    # ckpt_file = "tmp_logs/2023-05-08_13_39_31.158242/epoch=47-val_loss=0.12.ckpt"
    # ckpt_file = "lightning_logs/2023-05-07_13_34_19.194735/epoch=99-val_loss=0.14.ckpt"
    ckpt_file = "lightning_logs/2023-05-12_18_20_25.013834/l1000_ldm_con+vae_con_2023-05-12_18_20_25.013834/epoch=19-val_loss=0.31.ckpt"
    output_file = "cond_generation_res/ldm_con_vae_generated_molecules_and_sa_scores.pkl"
    mol_file = "cond_generation_res/ldm_con_vae_test_set_smile_to_max_sim_generated_molecule.pkl"
elif args.model_type == "aae":
    config_file = "config/ldm_con+aae_con.yml"
    # ckpt_file = "tmp_logs/2023-05-08_13_39_25.455320/epoch=47-val_loss=0.11.ckpt"
    # ckpt_file = "lightning_logs/2023-05-07_13_23_22.866645/epoch=95-val_loss=0.20.ckpt"
    ckpt_file = "lightning_logs/2023-05-12_18_20_27.525874/l1000_ldm_con+aae_con_2023-05-12_18_20_27.525874/epoch=20-val_loss=0.36.ckpt"
    output_file = "cond_generation_res/ldm_con_aae_generated_molecules_and_sa_scores.pkl"
    mol_file = "cond_generation_res/ldm_con_aae_test_set_smile_to_max_sim_generated_molecule.pkl"
elif args.model_type == "wae":
    config_file = "config/ldm_con+wae_con.yml"
    # ckpt_file = "tmp_logs/2023-05-08_13_39_19.133896/epoch=46-val_loss=0.13.ckpt"
    # ckpt_file = "lightning_logs/2023-05-07_13_23_05.773620/epoch=97-val_loss=0.09.ckpt"
    ckpt_file = "lightning_logs/2023-05-12_18_20_29.796081/l1000_ldm_con+wae_con_2023-05-12_18_20_29.796081/epoch=33-val_loss=0.19.ckpt"
    output_file = "cond_generation_res/ldm_con_wae_generated_molecules_and_sa_scores.pkl"
    mol_file = "cond_generation_res/ldm_con_wae_test_set_smile_to_max_sim_generated_molecule.pkl"
elif args.model_type == "test":
    config_file = "config/ldm_con+vae_con.yml"
    # ckpt_file = "tmp_logs/2023-05-08_13_39_31.158242/epoch=47-val_loss=0.12.ckpt"
    ckpt_file = "lightning_logs/2023-05-07_13_34_19.194735/epoch=99-val_loss=0.14.ckpt"
    output_file = "cond_generation_res/test_one.pkl"
    mol_file = "cond_generation_res/test_one.pkl"
else:
    raise ValueError("model type not supported")

config = OmegaConf.load(config_file)
ldm_params = config["model"]["params"]
first_stage_params = get_params(dataset)
first_stage_config = config['model']['first_stage_config']
ldm_params = config['model']['params']
unet_params = config['model']['unet_config']['params']
batch_size = 1
drop_prob = 0.0
latent_space_dim = int(ldm_params['image_size'])
size = [1, latent_space_dim]
device = torch.device(args.device)
# print("device: ", device)
if args.model_type == "aae" or args.model_type == "wae":
    first_stage_params["gene_exp_condition_mlp"]["input_feature_dim"] = 832 + 978 + 1

ckpt = torch.load(ckpt_file, map_location = device)
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
ldm_model.to(device=device)
ldm_model.eval()

results = {}

print("total number of test samples: ", len(reference_smiles))
# collect tensors into lists and then instantiate dataset
# i = 0
for control_idx, tumour_idx, reference_smile, original_idx in tqdm(
    zip(control_idxes, tumour_idxes, reference_smiles, original_idxes)
):
    # print("progress: ", i)
    # if i >= 100:
    #     break
    # print("evaluating ", original_idx)
    try:
        candidate_molecules = generate_similar_molecules_with_gene_exp_diff(
            control_idx,
            tumour_idx,
            original_idx,
            dataset,
            ldm_model,
            device,
            rand_vect_dim=512,
            num_samples=100,
        )
        results["_".join([reference_smile, str(original_idx)])] = {}
        results["_".join([reference_smile, str(original_idx)])]["generated_mols"] = [mol for mol in candidate_molecules]
        results["_".join([reference_smile, str(original_idx)])][
            "generated_smiles"
        ] = [Chem.MolToSmiles(mol) for mol in candidate_molecules]

        # candidate_molecules = [Chem.SanitizeMol(mol) for mol in candidate_molecules]
        for mol in candidate_molecules:
            Chem.SanitizeMol(mol)
        sa_scores = [sascorer.calculateScore(mol) for mol in candidate_molecules]
        results["_".join([reference_smile, str(original_idx)])][
            "sa_scores"
        ] = sa_scores
        # i += 1
    except Exception as e:
        print(e)

with open(output_file, "wb") as f:
    pickle.dump(results, f)

generated_mol_sims = {}
for reference_smile_original_idx in tqdm(results):
    try:
        reference_smile = reference_smile_original_idx.rsplit("_", 1)[0]
        # print("reference smile: ", reference_smile)
        # print("relevant result:", results[reference_smile_original_idx])
        # reference_smile = reference_smile.rsplit("_", 1)[0]
        max_sim = compute_max_similarity(
            # candidate_molecules=[
            #     Chem.MolFromSmiles(smile)
            #     for smile in results[reference_smile_original_idx][
            #         "generated_smiles"
            #     ]
            # ],
            candidate_molecules=results[reference_smile_original_idx]["generated_mols"],
            reference_smile=reference_smile,
        )
        generated_mol_sims[reference_smile_original_idx] = max_sim

    except Exception as e:
        # print(e)
        pass

with open(mol_file, "wb") as f:
    pickle.dump(generated_mol_sims, f)
