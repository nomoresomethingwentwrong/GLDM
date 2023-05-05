import sys 
sys.path.append('../')
import numpy as np
from dataset import LincsDataset
from torch_geometric.loader import DataLoader
import torch
from omegaconf import OmegaConf
from model_utils import get_params
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from moler_ldm import LatentDiffusion
import argparse

def filter_dataset(remove_idx, dataset):
    mask = np.ones(len(dataset), dtype=bool)
    mask[remove_idx] = False
    dataset = dataset[mask]
    return dataset

if __name__ == "__main__":

    batch_size = 1
    NUM_WORKERS = 4
    train_split1 = "train_0"
    valid_split = "valid_0"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layer_type",
        required=True,
        type=str,
        choices=["FiLMConv", "GATConv", "GCNConv"],
    )
    parser.add_argument(
        "--model_architecture", required=True, type=str, choices=["aae", "vae"]
    )
    parser.add_argument("--use_oclr_scheduler", action="store_true")
    parser.add_argument("--using_cyclical_anneal", action="store_true")
    parser.add_argument("--using_wasserstein_loss", action="store_true")
    parser.add_argument("--use_clamp_log_var", action="store_true")
    parser.add_argument("--using_gp", action="store_true")
    parser.add_argument("--gradient_clip_val", required=True, type=float, default=1.0)
    parser.add_argument("--max_lr", required=True, type=float, default=1e-5)
    parser.add_argument(
        "--gen_step_drop_probability", required=True, type=float, default=0.5
    )
    parser.add_argument("--pretrained_ckpt", type=str)
    parser.add_argument("--pretrained_ckpt_model_type", type=str)
    parser.add_argument("--config_file", type=str, default="config/ddim_vae_uncon.yml")

    '''
    VAE: 
    python train_ldm.py --layer_type=FiLMConv --model_architecture=vae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --gen_step_drop_probability=0
    '''

    args = parser.parse_args()

    raw_moler_trace_dataset_parent_folder = "/data/ongh0068/guacamol/trace_dir"
    output_pyg_trace_dataset_parent_folder = (
        "/data/ongh0068/l1000/already_batched"
    )

    config = OmegaConf.load(args.config_file)
    ldm_params = config['model']['params']

    train_dataset = LincsDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        gene_exp_controls_file_path="/data/ongh0068/l1000/lincs/robust_normalized_controls.npz",
        gene_exp_tumour_file_path="/data/ongh0068/l1000/lincs/robust_normalized_tumors.npz",
        lincs_csv_file_path="/data/ongh0068/l1000/lincs/experiments_filtered.csv",
        split=train_split1,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    valid_dataset = LincsDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        gene_exp_controls_file_path="/data/ongh0068/l1000/lincs/robust_normalized_controls.npz",
        gene_exp_tumour_file_path="/data/ongh0068/l1000/lincs/robust_normalized_tumors.npz",
        lincs_csv_file_path="/data/ongh0068/l1000/lincs/experiments_filtered.csv",
        split=valid_split,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    # train_dataset = filter_dataset(791, train_dataset)
    # valid_dataset = filter_dataset(791, valid_dataset)

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
        num_workers=NUM_WORKERS,
        # prefetch_factor=0,
    )

    valid_dataset = valid_dataset[:200]  # use only 200 batches for validation
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
        num_workers=NUM_WORKERS,
        # prefetch_factor=0,
    )

    # print(len(train_dataloader), len(valid_dataloader))
    # print(next(iter(train_dataloader)))
    first_stage_params = get_params(train_dataset)
    ###################################################
    first_stage_params["full_graph_encoder"]["layer_type"] = args.layer_type
    first_stage_params["partial_graph_encoder"]["layer_type"] = args.layer_type
    first_stage_params["use_oclr_scheduler"] = args.use_oclr_scheduler
    first_stage_params["using_cyclical_anneal"] = args.using_cyclical_anneal
    model_architecture = args.model_architecture
    first_stage_params["max_lr"] = args.max_lr
    ###################################################
    first_stage_config = config['model']['first_stage_config']

    if model_architecture == "aae":
        first_stage_params["gene_exp_condition_mlp"]["input_feature_dim"] = 832 + 978 + 1

    ldm_model = LatentDiffusion(
        first_stage_config,
        config['model']['cond_stage_config'],
        train_dataset, 
        batch_size,
        first_stage_params,
        first_stage_config['ckpt_path'],
        unet_config = config['model']['unet_config'],
        **ldm_params
    )

    lr = config.model.base_learning_rate
    ldm_model.learning_rate = lr    

    # Get current time for folder path.
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=f"lightning_logs/{now}", name=f"logs_{now}")
    early_stopping = EarlyStopping(monitor=ldm_params.monitor, patience=10)
    if model_architecture == "vae":
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
            dirpath=f"lightning_logs/{now}",
            mode="min",
            filename='epoch{epoch:02d}-dropout{args.gen_step_drop_probability:.2f}-val_loss{val/loss:.2f}',
        )
    elif model_architecture == "aae":
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"lightning_logs/{now}",
            filename="{epoch:02d}-{train_loss:.2f}",
            monitor="epoch",
            every_n_epochs=3,
            save_on_train_epoch_end=True,
            save_top_k=-1,
        )
    else:
        raise NotImplementedError('model_architecture must be either "vae" or "aae"')

    callbacks = (
        [checkpoint_callback, lr_monitor, early_stopping]
        if model_architecture == "vae"
        else [checkpoint_callback, lr_monitor]
    )

    trainer = Trainer(accelerator='gpu', 
                      max_epochs=100, 
                    #   num_sanity_val_steps=0,    # the CUDA capability is insufficient to train the whole batch, we drop some graphs in each batch, but need to set num_sanity_val_steps=0 to avoid the validation step to run (with the whole batch)
                      devices=[2], 
                      callbacks=callbacks, 
                      logger=tensorboard_logger, 
                      gradient_clip_val=args.gradient_clip_val)
    trainer.fit(ldm_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)