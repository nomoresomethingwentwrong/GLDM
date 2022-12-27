from dataset import MolerDataset
from torch_geometric.loader import DataLoader
from model import BaseModel
from model_utils import get_params
from torch.utils.data import RandomSampler
from sampler import DuplicatedIndicesSamplerWrapper
import pandas as pd
from pytorch_lightning import Trainer


# if __name__ == '__main__':

print('here')

train_dataset = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder='/data/ongh0068/guacamol/trace_dir',#"/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/pyg_output_playground",
    split="train_0",
)

# processed_file_metadata = (
#     "/data/ongh0068/l1000/pyg_output_playground/train/processed_file_paths.csv"
# )
# molecule_gen_steps_lengths = pd.read_csv(processed_file_metadata)[
#     "molecule_gen_steps_length"
# ].tolist()

# random_sampler = RandomSampler(data_source=[i for i in range(len(train_dataset))])
# sampler = DuplicatedIndicesSamplerWrapper(
#     sampler=random_sampler,
#     frequency_mapping={
#         idx: length for idx, length in enumerate(molecule_gen_steps_lengths)
#     },
# )


# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=256,
#     shuffle=False,
#     sampler=sampler,
#     follow_batch=[
#         "correct_edge_choices",
#         "correct_edge_types",
#         "valid_edge_choices",
#         "valid_attachment_point_choices",
#         "correct_attachment_point_choice",
#         "correct_node_type_choices",
#         "original_graph_x",
#         'correct_first_node_type_choices'
#     ],
# )

# params = get_params()
# model = BaseModel(params, train_dataset)


# trainer = Trainer(accelerator = 'gpu', max_epochs = 100, devices = 1)  # overfit_batches=1)
# trainer.fit(model, train_dataloader, train_dataloader)