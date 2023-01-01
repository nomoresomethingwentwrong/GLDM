from dataset import MolerDataset
from torch_geometric.loader import DataLoader
from model import BaseModel
from model_utils import get_params
from torch.utils.data import RandomSampler
from sampler import DuplicatedIndicesSamplerWrapper
import pandas as pd
from pytorch_lightning import Trainer


if __name__ == '__main__':

    # train_split = 'train_0'
    valid_split = 'valid_0'

    # train_dataset = MolerDataset(
    #     root="/data/ongh0068",
    #     raw_moler_trace_dataset_parent_folder='/data/ongh0068/guacamol/trace_dir',#"/data/ongh0068/l1000/trace_playground",
    #     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/pyg_output_playground",
    #     split=train_split,
    # )
    raw_moler_trace_dataset_parent_folder = '/data/ongh0068/guacamol/trace_dir'
    output_pyg_trace_dataset_parent_folder = "/data/ongh0068/l1000/already_batched"

    valid_dataset = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,#"/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=valid_split,
    )


    # train_processed_file_metadata = (
    #     f"/data/ongh0068/l1000/pyg_output_playground/{train_split}/processed_file_paths.csv"
    # )
    # train_molecule_gen_steps_lengths = pd.read_csv(train_processed_file_metadata)[
    #     "molecule_gen_steps_length"
    # ].tolist()
    # train_random_sampler = RandomSampler(data_source=[i for i in range(len(train_dataset))])
    # train_sampler = DuplicatedIndicesSamplerWrapper(
    #     sampler=train_random_sampler,
    #     frequency_mapping={
    #         idx: length for idx, length in enumerate(train_molecule_gen_steps_lengths)
    #     },
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=16,
    #     shuffle=False,
    #     sampler=train_sampler,
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

    valid_processed_file_metadata = (
        f"{output_pyg_trace_dataset_parent_folder}/{valid_split}/processed_file_paths.csv"
    )
    # valid_molecule_gen_steps_lengths = pd.read_csv(valid_processed_file_metadata)[
    #     "molecule_gen_steps_length"
    # ].tolist()
    # valid_random_sampler = RandomSampler(data_source=[i for i in range(len(valid_dataset))])
    # valid_sampler = DuplicatedIndicesSamplerWrapper(
    #     sampler=valid_random_sampler,
    #     frequency_mapping={
    #         idx: length for idx, length in enumerate(valid_molecule_gen_steps_lengths)
    #     },
    # )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        # sampler=valid_sampler,
        follow_batch=[
            "correct_edge_choices",
            "correct_edge_types",
            "valid_edge_choices",
            "valid_attachment_point_choices",
            "correct_attachment_point_choice",
            "correct_node_type_choices",
            "original_graph_x",
            'correct_first_node_type_choices'
        ],
    )




    params = get_params(dataset=valid_dataset)#train_dataset)
    model = BaseModel(params, valid_dataset)#train_dataset)


    trainer = Trainer(accelerator = 'gpu', max_epochs = 100, devices = [2])  # overfit_batches=1)
    trainer.fit(model, valid_dataloader, valid_dataloader)