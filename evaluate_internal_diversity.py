from l1000_evaluation_utils import internal_diversity
import pickle
from tqdm import tqdm
import os
import json
from rdkit import Chem
import pandas as pd
# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/fs_vae_generated_molecules_and_sa_scores.pkl",
#     "rb",
# ) as f:
#     fs_vae_data = pickle.load(f)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/tl_vae_generated_molecules_and_sa_scores2.pkl",
#     "rb",
# ) as f:
#     tl_vae_data = pickle.load(f)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/fs_aae_generated_molecules_and_sa_scores.pkl",
#     "rb",
# ) as f:
#     fs_aae_data = pickle.load(f)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/tl_aae_generated_molecules_and_sa_scores2.pkl",
#     "rb",
# ) as f:
#     tl_aae_data = pickle.load(f)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/fs_wae_generated_molecules_and_sa_scores.pkl",
#     "rb",
# ) as f:
#     fs_wae_data = pickle.load(f)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/tl_wae_generated_molecules_and_sa_scores2.pkl",
#     "rb",
# ) as f:
#     tl_wae_data = pickle.load(f)


# def extract_mols(data):
#     tmp = []
#     for test_set_smile in data:
#         tmp += [
#             Chem.MolFromSmiles(smile)
#             for smile in data[test_set_smile]["generated_smiles"]
#         ]
#     return tmp


# l1000_smiles = {
#     "tl_vae": extract_mols(tl_vae_data),
#     "fs_vae": extract_mols(fs_vae_data),
#     "tl_aae": extract_mols(tl_aae_data),
#     "fs_aae": extract_mols(fs_aae_data),
#     "fs_wae": extract_mols(fs_wae_data),
#     "tl_wae": extract_mols(tl_wae_data),
# }

# internal_diversity_results_l1000 = {}
# for config in tqdm(l1000_smiles):
#     try:
#         internal_diversity_results_l1000[config] = internal_diversity(
#             l1000_smiles[config]
#         )
#     except Exception as e:
#         print(e)
# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/internal_diversity/l1000_results.pkl",
#     "wb",
# ) as f:
#     pickle.dump(internal_diversity_results_l1000, f)


# internal_diversity_results_guac = {}

# for json_file in tqdm(
#     os.listdir(
#         "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/distribution_learning_benchmark"
#     )
# ):
#     try:
#         full_json_file = os.path.join(
#             "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/distribution_learning_benchmark",
#             json_file,
#         )
#         f = open(full_json_file)
#         data = json.load(f)
#         mols = [Chem.MolFromSmiles(smile) for smile in data['samples']]
#         internal_diversity_results_guac[json_file] = internal_diversity(mols)
#     except Exception as e:
#         print(e)

# with open(
#     "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/internal_diversity/guac_results.pkl",
#     "wb",
# ) as f:
#     pickle.dump(internal_diversity_results_guac, f)


internal_diversity_results_original_l1000 = {}

df = pd.read_csv('/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv')

smiles = df.SMILES.unique().tolist()

mols = [Chem.MolFromSmiles(smile) for smile in smiles]
internal_diversity_results_original_l1000['original_smiles'] = internal_diversity(mols)
with open(
    "/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/internal_diversity/original_l1000.pkl",
    "wb",
) as f:
    pickle.dump(internal_diversity_results_original_l1000, f)
