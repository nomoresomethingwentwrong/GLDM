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
    parser.add_argument(
        "--ckpt_file_path",
        default="/data/ongh0068/2023-02-04_20_40_45.735930/epoch=06-val_loss=0.47.ckpt",
    )
    parser.add_argument("--layer_type")
    parser.add_argument("--model_type")
    parser.add_argument("--using_lincs", type=bool)
    parser.add_argument("--output_fp", default="distribution_learning_results.json")
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
    )

    json_file_path = os.path.join(args.output_dir, "distribution_learning_results.json")

    assess_distribution_learning(
        generator,
        chembl_training_file=args.dist_file,
        json_output_file=json_file_path,
        benchmark_version=args.suite,
    )
