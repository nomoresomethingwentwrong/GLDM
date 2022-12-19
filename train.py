from dataset import MolerDataset, MolerData
from utils import pprint_pyg_obj
from torch_geometric.loader import DataLoader
from model import BaseModel
from model_utils import get_params

from pytorch_lightning import Trainer
import torch

train_dataset = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder="/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/pyg_output_playground",
    split="train",
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    follow_batch=[
        "correct_edge_choices",
        "correct_edge_types",
        "valid_edge_choices",
        "valid_attachment_point_choices",
        "correct_attachment_point_choice",
        "correct_node_type_choices",
        "original_graph_x",
    ],
)
params = get_params()
model = BaseModel(params, train_dataset).to("cuda:1")


# datamodule = LightningDataset(dataset)
trainer = Trainer()  # overfit_batches=1)
trainer.fit(model, train_dataloader, train_dataloader)
