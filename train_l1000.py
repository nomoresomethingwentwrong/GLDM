from dataset import LincsDataset
from torch_geometric.loader import DataLoader
from model import BaseModel
from aae import AAE
from model_utils import get_params
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import sys

"""python train_l1000.py FiLMConv vae"""


if __name__ == "__main__":

    batch_size = 1
    train_split1 = "train_0"
    valid_split = "valid_0"

    raw_moler_trace_dataset_parent_folder = "/data/ongh0068/guacamol/trace_dir"
    output_pyg_trace_dataset_parent_folder = (
        "/data/ongh0068/l1000/l1000_biaae/already_batched"
    )

    train_dataset = LincsDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
        gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
        lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
        split=train_split1,
    )

    valid_dataset = LincsDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
        gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
        lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
        split=valid_split,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # sampler=train_sampler,
        follow_batch=[
            "correct_edge_choices",
            "correct_edge_types",
            "valid_edge_choices",
            "valid_attachment_point_choices",
            "correct_attachment_point_choice",
            "correct_node_type_choices",
            "original_graph_x",
            "correct_first_node_type_choices",
        ],
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
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
            "correct_first_node_type_choices",
        ],
    )

    params = get_params(dataset=train_dataset)  # train_dataset)
    ###################################################
    layer_type = sys.argv[1]  # change this
    params["full_graph_encoder"]["layer_type"] = layer_type
    params["partial_graph_encoder"]["layer_type"] = layer_type
    # params['using_cyclical_anneal'] = True
    model_architecture = sys.argv[2]  # expects aae, vae
    if model_architecture == "aae":
        model = AAE(
            params,
            valid_dataset,
            using_lincs=True,
            num_train_batches=len(train_dataloader),
            batch_size=batch_size,
        )
    elif model_architecture == "vae":
        model = BaseModel(
            params,
            valid_dataset,
            using_lincs=True,
            num_train_batches=len(train_dataloader),
            batch_size=batch_size,
        )  # train_dataset)
    else:
        raise ValueError
    ###################################################

    # Get current time for folder path.
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=f"../{now}", name=f"logs_{now}")
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    if model_architecture == "vae":
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            dirpath=f"../{now}",
            mode="min",
            filename="{epoch:02d}-{val_loss:.2f}",
        )
    elif model_architecture == "aae":
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"../{now}",
            filename="{epoch:02d}-{train_loss:.2f}",
            every_n_epochs=2,
        )

    callbacks = (
        [checkpoint_callback, lr_monitor, early_stopping]
        if model_architecture == "vae"
        else [checkpoint_callback, lr_monitor]
    )
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=30,
        devices=[2],
        callbacks=callbacks,
        logger=tensorboard_logger,
        gradient_clip_val=1.0,
        # detect_anomaly=True,
        # track_grad_norm=int(sys.argv[3]), # set to 2 for l2 norm
    )  # overfit_batches=1)
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
