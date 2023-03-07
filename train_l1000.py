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
from model_utils import transfer_trained_weights
import argparse

# import torch.multiprocessing as mp

"""python train_l1000.py FiLMConv vae"""


if __name__ == "__main__":

    batch_size = 1
    NUM_WORKERS = 4
    train_split1 = "train_0"
    valid_split = "valid_0"
    parser = argparse.ArgumentParser()
    """
    pretrained_ckpt can be None, in which case transfer learning won't be used.

    python train_l1000.py \
    --layer_type=FiLMConv \
    --model_architecture=vae \
    --use_oclr_scheduler \
    --using_cyclical_anneal \
    --gradient_clip_val=1.0 \
    --max_lr=1e-4 \
    --gen_step_drop_probability=0.5 \
    --pretrained_ckpt=/data/ongh0068/l1000/2023-03-03_09_30_01.589479/epoch=12-val_loss=0.46.ckpt \
    --pretrained_ckpt_model_type=vae \
    """
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
    parser.add_argument("--using_gp", action="store_true")
    parser.add_argument("--gradient_clip_val", required=True, type=float, default=1.0)
    parser.add_argument("--max_lr", required=True, type=float, default=1e-5)
    parser.add_argument(
        "--gen_step_drop_probability", required=True, type=float, default=0.5
    )
    parser.add_argument("--pretrained_ckpt", required=True, type=str)
    parser.add_argument("--pretrained_ckpt_model_type", required=True, type=str)

    args = parser.parse_args()

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
        gen_step_drop_probability=args.gen_step_drop_probability,
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
        num_workers=NUM_WORKERS,
        # prefetch_factor=0,
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
        num_workers=NUM_WORKERS,
        # prefetch_factor=0,
    )
    print(len(train_dataloader), len(valid_dataloader))

    params = get_params(dataset=train_dataset)  # train_dataset)
    ###################################################
    params["full_graph_encoder"]["layer_type"] = args.layer_type
    params["partial_graph_encoder"]["layer_type"] = args.layer_type
    params["use_oclr_scheduler"] = args.use_oclr_scheduler
    params["using_cyclical_anneal"] = args.using_cyclical_anneal
    model_architecture = args.model_architecture
    params["max_lr"] = args.max_lr
    ###################################################

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

    if args.pretrained_ckpt is not None:
        print(f"Transfering weights from {args.pretrained_ckpt}...")
        assert args.pretrained_ckpt_model_type is not None
        if args.pretrained_ckpt_model_type == "vae":
            pretrained_model = BaseModel.load_from_checkpoint(
                args.pretrained_ckpt,
                params=params,
                dataset=valid_dataset,
                using_lincs=False,
                num_train_batches=len(train_dataloader),
                batch_size=batch_size,
            )
        elif args.pretrained_ckpt_model_type == "aae":
            pretrained_model = AAE.load_from_checkpoint(
                args.pretrained_ckpt,
                params=params,
                dataset=valid_dataset,
                using_lincs=False,
                num_train_batches=len(train_dataloader),
                batch_size=batch_size,
            )
        else:
            raise ValueError
        transfer_trained_weights(pretrained_model, model)
        del pretrained_model
        print("Done transfering weights")
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
            monitor="epoch",
            every_n_epochs=3,
            save_on_train_epoch_end=True,
            save_top_k=-1,
        )

    callbacks = (
        [checkpoint_callback, lr_monitor, early_stopping]
        if model_architecture == "vae"
        else [checkpoint_callback, lr_monitor]
    )
    # mp.set_start_method('spawn')
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=30,
        devices=[2],
        callbacks=callbacks,
        logger=tensorboard_logger,
        gradient_clip_val=args.gradient_clip_val,
        # fast_dev_run=True
        # detect_anomaly=True,
        # track_grad_norm=int(sys.argv[3]), # set to 2 for l2 norm
    )  # overfit_batches=1)
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
