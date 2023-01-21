
from model import BaseModel
from dataset import MolerDataset, MolerData
from utils import pprint_pyg_obj
from torch_geometric.loader import DataLoader
import torch
from model_utils import get_params

from torch.utils.data import ConcatDataset


# dataset = MolerDataset(
#     root = '/data/ongh0068', 
#     raw_moler_trace_dataset_parent_folder = '/data/ongh0068/guacamol/trace_dir',
#     output_pyg_trace_dataset_parent_folder = '/data/ongh0068/l1000/already_batched',
#     split = 'train_0',
# )
train_split1 = "train_0"
train_split2 = "train_1000"
train_split3 = "train_2000"
train_split4 = "train_3000"
train_split5 = "train_4000"
train_split6 = "train_5000"
train_split7 = "train_6000"

raw_moler_trace_dataset_parent_folder = "/data/ongh0068/guacamol/trace_dir"
output_pyg_trace_dataset_parent_folder = "/data/ongh0068/l1000/already_batched"

train_dataset1 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split1,
)
train_dataset2 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split2,
)
train_dataset3 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split3,
)

train_dataset4 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split4,
)

train_dataset5 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split5,
)
train_dataset6 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split6,
)
train_dataset7 = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
    split=train_split7,
)
train_dataset = ConcatDataset(
    [
        train_dataset1,
        train_dataset2,
        train_dataset3,
        train_dataset4,
        train_dataset5,
        train_dataset6,
        train_dataset7,
    ]
)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False, follow_batch = [
    'correct_edge_choices',
    'correct_edge_types',
    'valid_edge_choices',
    'valid_attachment_point_choices',
    'correct_attachment_point_choice',
    'correct_node_type_choices',
    'original_graph_x',
    'correct_first_node_type_choices'
])


params = get_params(train_dataset1)


ckpt_file_path = '/data/ongh0068/l1000/2023-01-18_20_53_14.078174/epoch=08-val_loss=3.69.ckpt'
# '/data/ongh0068/l1000/2023-01-09_13_46_22.653611/epoch=48-val_loss=3.06.ckpt'
# '/data/ongh0068/l1000/2023-01-09_13_46_22.653611/epoch=07-val_loss=4.40.ckpt'
# '/data/ongh0068/l1000/2023-01-08_07_51_26.751380/epoch=14-val_loss=6.56.ckpt'
# '/data/ongh0068/l1000/2023-01-08_07_51_26.751380/epoch=05-val_loss=6.62.ckpt'
# '/data/ongh0068/l1000/2023-01-08_07_51_26.751380/epoch=02-val_loss=7.12.ckpt'
# '/data/ongh0068/l1000/2023-01-07_07_24_40.584868/epoch=15-val_loss=7.52.ckpt'
#'/data/ongh0068/l1000/2023-01-07_14_31_37.584989/epoch=00-val_loss=10.90.ckpt'
# '/data/ongh0068/l1000/2023-01-06_16_41_53.821231/epoch=00-val_loss=10.60.ckpt'
# '/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/lightning_logs/version_10/checkpoints/epoch=47-step=888960.ckpt'
# '/data/ongh0068/l1000/2023-01-06_12_10_45.376222/epoch=00-val_f1=0.00.ckpt'
# '/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/lightning_logs/version_10/checkpoints/epoch=47-step=888960.ckpt'
# '/data/ongh0068/l1000/FYP-DrugDiscoveryWithDeepLearning/lightning_logs/version_8/checkpoints/epoch=99-step=245600.ckpt'

model = BaseModel.load_from_checkpoint(ckpt_file_path, params = params, dataset = train_dataset1)
model.eval()

decoder_states = []
from tqdm import tqdm
for batch in tqdm(loader):
    with torch.no_grad():
        input_molecule_representations = model._full_graph_encoder(
            original_graph_node_categorical_features=batch.original_graph_node_categorical_features,
            node_features=batch.original_graph_x.float(),
            edge_index=batch.original_graph_edge_index,
            edge_features=batch.original_graph_edge_features.float(),
            batch_index=batch.original_graph_x_batch,
        )
        p, q, latent_representations = model.sample_from_latent_repr(
            input_molecule_representations
        )
    decoder_states += model.decode(latent_representations = latent_representations, max_num_steps = 120)


from rdkit.Chem import Draw
from rdkit import Chem
mols = [decoder_states[i].molecule for i in range(len(decoder_states))]
smiles = [Chem.MolToSmiles(m) for m in mols]
unique_smiles = list(set(smiles))
unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
img = Draw.MolsToGridImage(unique_mols[-100:], subImgSize=(200,200), maxMols = 100, molsPerRow=5)

import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(unique_smiles)

with open("recover_training_set.png", "wb") as png:
    png.write(img.data)