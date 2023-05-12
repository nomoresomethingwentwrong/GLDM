import sys 
sys.path.append('../')
import numpy as np
from dataset import MolerDataset
from torch_geometric.loader import DataLoader
import torch
from omegaconf import OmegaConf
from model_utils import get_params
from torch.utils.data import ConcatDataset
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
    # train_split1 = "train_0"
    # valid_split = "valid_0"
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
    parser.add_argument("--config_file", type=str, default="config/ldm_uncon+vae_uncon.yml")

    '''
    VAE (unconditional): 
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=vae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --gen_step_drop_probability=0.9 --config_file=config/ldm_uncon+vae_uncon.yml

    AAE (unconditional):
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=aae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --gen_step_drop_probability=0.9 --config_file=config/ldm_uncon+aae_uncon.yml

    WAE (unconditional):
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=aae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --using_wasserstein_loss --using_gp --gen_step_drop_probability=0.9 --config_file=config/ldm_uncon+wae_uncon.yml

    VAE (conditional): 
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=vae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --gen_step_drop_probability=0.9 --config_file=config/ldm_con+vae_con.yml

    AAE (conditional)):
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=aae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --gen_step_drop_probability=0.95 --config_file=config/ldm_con+aae_con.yml

    WAE (conditional):
    python train_ldm_guacamol.py --layer_type=FiLMConv --model_architecture=aae --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 --using_wasserstein_loss --using_gp --gen_step_drop_probability=0.9 --config_file=config/ldm_con+wae_con.yml
    '''

    args = parser.parse_args()

    train_split1 = "train_0"
    train_split2 = "train_1000"
    train_split3 = "train_2000"
    train_split4 = "train_3000"
    train_split5 = "train_4000"
    train_split6 = "train_5000"
    train_split7 = "train_6000"

    valid_split = "valid_0"

    raw_moler_trace_dataset_parent_folder = "/data/ongh0068/guacamol/trace_dir"
    output_pyg_trace_dataset_parent_folder = (
        "../data/guacamol/already_batched"
    )

    config = OmegaConf.load(args.config_file)
    ldm_params = config['model']['params']

    train_dataset1 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split1,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )
    train_dataset2 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split2,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )
    train_dataset3 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split3,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    train_dataset4 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split4,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    train_dataset5 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split5,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )
    train_dataset6 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split6,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )
    train_dataset7 = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=train_split7,
        gen_step_drop_probability=args.gen_step_drop_probability,
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

    valid_dataset = MolerDataset(
        root="/data/ongh0068",
        raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # "/data/ongh0068/l1000/trace_playground",
        output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,
        split=valid_split,
        gen_step_drop_probability=0.0
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
        num_workers=NUM_WORKERS,
    )

    valid_dataset = valid_dataset[:100]  # use only 100 batches for validation
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
    )

    # print(len(train_dataloader), len(valid_dataloader))
    # print(next(iter(train_dataloader)))
    first_stage_params = get_params(train_dataset1)
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
        valid_dataset, 
        args.gen_step_drop_probability,
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
    early_stopping = EarlyStopping(monitor=ldm_params.monitor, patience=3)
    if model_architecture == "vae" or model_architecture == "aae":
        # checkpoint_callback = ModelCheckpoint(
        #     save_top_k=1,
        #     monitor="val/loss",
        #     dirpath=f"lightning_logs/{now}",
        #     mode="min",
        #     filename='epoch={epoch:02d}-val_loss={val/loss:.2f}',
        #     auto_insert_metric_name=False,
        # )
        checkpoint_callback = ModelCheckpoint(
            # save_top_k=1,
            monitor="val/loss",
            dirpath=f"lightning_logs/{now}",
            # mode="min",
            every_n_train_steps=3000,
            filename='epoch={epoch:02d}-step={global_step}-val_loss={val/loss:.2f}',
            auto_insert_metric_name=False,
        )
    # elif model_architecture == "aae":
    #     checkpoint_callback = ModelCheckpoint(
    #         dirpath=f"lightning_logs/{now}",
    #         filename="{epoch:02d}-{train_loss:.2f}",
    #         monitor="epoch",
    #         every_n_epochs=3,
    #         save_on_train_epoch_end=True,
    #         save_top_k=-1,
    #     )
    else:
        raise NotImplementedError('model_architecture must be either "vae" or "aae"')

    callbacks = (
        [checkpoint_callback, lr_monitor, early_stopping]
        # if model_architecture == "vae"
        # else [checkpoint_callback, lr_monitor]
    )

    trainer = Trainer(accelerator='gpu', 
                      max_epochs=100, 
                    #   num_sanity_val_steps=0,    # the CUDA capability is insufficient to train the whole batch, we drop some graphs in each batch, but need to set num_sanity_val_steps=0 to avoid the validation step to run (with the whole batch)
                      devices=[0], 
                      callbacks=callbacks, 
                      logger=tensorboard_logger, 
                      gradient_clip_val=args.gradient_clip_val)
    trainer.fit(ldm_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)