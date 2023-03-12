import os
import argparse
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger
from evaluation_utils import MoLeRGenerator

if __name__ == "__main__":
    setup_default_logger()
    """
    --dist_file=/data/ongh0068/l1000/l1000_biaae/lincs/l1000.smiles
    python distribution_learning.py  --ckpt_file_path=/data/ongh0068/l1000/2023-03-03_09_26_09.229843/epoch=03-train_loss=0.00.ckpt --layer_type=FiLMConv --model_type=aae --using_lincs=False --output_dir=distribution_learning_benchmark --output_fp=aae_ep3_distribution_learning_results.json
    python distribution_learning.py  --ckpt_file_path=/data/ongh0068/l1000/2023-03-01_13_40_54.126319/epoch=28-val_loss=0.39.ckpt --layer_type=FiLMConv --model_type=vae --using_lincs=True

    # WAE + GP + no oclr + no genstep_drop (epoch 5)
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-08_07_54_03.497887/epoch=05-train_loss=0.28.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --using_wasserstein_loss --using_gp \
        --output_dir=distribution_learning_benchmark \
        --output_fp=wae_no_oclr_no_genstep_ep5_distribution_learning_results.json

    # WAE + GP + oclr + no genstep_drop
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-07_23_24_09.367132/epoch=05-train_loss=0.23.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --using_wasserstein_loss --using_gp \
        --output_dir=distribution_learning_benchmark \
        --output_fp=wae_oclr_no_genstep_drop_ep5_distribution_learning_results.json

    # WAE + GP + no oclr + no genstep_drop (epoch 8)
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-08_07_54_03.497887/epoch=08-train_loss=0.10.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --using_wasserstein_loss --using_gp \
        --output_dir=distribution_learning_benchmark \
        --output_fp=wae_no_oclr_no_genstep_drop_ep8_distribution_learning_results.json

    # AAE + no oclr + no gen step drop (Vanilla)
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-06_19_22_54.162250/epoch=08-train_loss=0.70.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --output_dir=distribution_learning_benchmark \
        --output_fp=aae_no_oclr_no_genstep_drop_distribution_learning_results.json

    # VAE + oclr + kl anneal + gen step drop 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-05_14_24_55.916122/epoch=24-val_loss=0.29.ckpt \
        --layer_type=FiLMConv \
        --model_type=vae \
        --output_dir=distribution_learning_benchmark \
        --output_fp=vae_oclr_kl_anneal_genstep_drop_distribution_learning_results.json \
        --device='cuda:1'

    # VAE + no oclr + kl anneal + gen step drop 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-06_19_02_43.513099/epoch=15-val_loss=0.38.ckpt \
        --layer_type=FiLMConv \
        --model_type=vae \
        --output_dir=distribution_learning_benchmark \
        --output_fp=vae_no_oclr_kl_anneal_genstep_drop_distribution_learning_results.json \
        --device='cuda:3'

    # AAE + oclr + gen step drop 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-06_16_47_15.554929/epoch=11-train_loss=0.71.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --output_dir=distribution_learning_benchmark \
        --output_fp=aae_oclr_genstep_drop_distribution_learning_results.json \
        --device='cuda:3'  

    # AAE + no oclr + gen step drop 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-06_19_00_13.612395/epoch=11-train_loss=0.68.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --output_dir=distribution_learning_benchmark \
        --output_fp=aae_no_oclr_genstep_drop_distribution_learning_results.json \
        --device='cuda:1'  

    #########
    # Transfer Learning VAE oclr + kl anneal + vae 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-06_11_35_00.377565/epoch=12-val_loss=0.05.ckpt \
        --layer_type=FiLMConv \
        --model_type=vae \
        --using_lincs \
        --output_dir=distribution_learning_benchmark \
        --output_fp=tl_l1000_vae_no_oclr_distribution_learning_results.json
    #########

    # Transfer Learning VAE 
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-11_20_37_28.240239/epoch=05-val_loss=0.63.ckpt \
        --layer_type=FiLMConv \
        --model_type=vae \
        --using_lincs \
        --output_dir=distribution_learning_benchmark \
        --output_fp=tl_l1000_vae_best_distribution_learning_results.json

    # Transfer Learning VAE Lower learning rate
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-11_23_33_36.921147/epoch=07-val_loss=0.60.ckpt \
        --layer_type=FiLMConv \
        --model_type=vae \
        --using_lincs \
        --output_dir=distribution_learning_benchmark \
        --output_fp=tl_l1000_vae_best_lower_lr_distribution_learning_results.json

    # Transfer Learning WAE
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-11_20_54_14.382629/epoch=08-train_loss=-0.39.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --using_lincs \
        --using_wasserstein_loss --using_gp \
        --output_dir=distribution_learning_benchmark \
        --output_fp=tl_l1000_wae_best_distribution_learning_results.json


    # Transfer Learning AAE
    python distribution_learning.py  \
        --ckpt_file_path=/data/ongh0068/l1000/2023-03-11_20_54_15.863102/epoch=20-train_loss=0.00.ckpt \
        --layer_type=FiLMConv \
        --model_type=aae \
        --using_lincs \
        --output_dir=distribution_learning_benchmark \
        --output_fp=tl_l1000_aae_best_distribution_learning_results.json

    

    """
    parser = argparse.ArgumentParser(
        description="Molecule distribution learning benchmark for random smiles sampler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dist_file", default="/data/ongh0068/guacamol/guacamol_v1_all.smiles"
    )
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--suite", default="v2")
    parser.add_argument("--using_wasserstein_loss", action="store_true")
    parser.add_argument("--using_gp", action="store_true")
    parser.add_argument(
        "--ckpt_file_path",
        default="/data/ongh0068/2023-02-04_20_40_45.735930/epoch=06-val_loss=0.47.ckpt",
    )
    parser.add_argument("--layer_type")
    parser.add_argument("--model_type")
    parser.add_argument("--using_lincs", action="store_true")
    parser.add_argument("--output_fp", default="distribution_learning_results.json")
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(args.dist_file, "r") as smiles_file:
        smiles_list = [line.strip() for line in smiles_file.readlines()]

    generator = MoLeRGenerator(
        ckpt_file_path=args.ckpt_file_path,
        layer_type=args.layer_type,
        model_type=args.model_type,
        using_lincs=args.using_lincs,
        using_gp=True if args.using_gp else False,
        using_wasserstein_loss=True if args.using_wasserstein_loss else False,
        device=args.device,
    )

    json_file_path = os.path.join(args.output_dir, args.output_fp)

    assess_distribution_learning(
        generator,
        chembl_training_file=args.dist_file,
        json_output_file=json_file_path,
        benchmark_version=args.suite,
    )
