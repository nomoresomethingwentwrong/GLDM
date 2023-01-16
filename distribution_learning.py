import os
import argparse
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger
from evaluation_utils import MoLeRGenerator

if __name__ == '__main__':
    setup_default_logger()

    parser = argparse.ArgumentParser(description='Molecule distribution learning benchmark for random smiles sampler',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dist_file', default='/data/ongh0068/guacamol/guacamol_v1_all.smiles')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--ckpt_file_path', default = '/data/ongh0068/l1000/2023-01-09_13_46_22.653611/epoch=48-val_loss=3.06.ckpt')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(args.dist_file, 'r') as smiles_file:
        smiles_list = [line.strip() for line in smiles_file.readlines()]

    generator = MoLeRGenerator(ckpt_file_path=args.ckpt_file_path)

    json_file_path = os.path.join(args.output_dir, 'distribution_learning_results.json')

    assess_distribution_learning(generator,
                                 chembl_training_file=args.dist_file,
                                 json_output_file=json_file_path,
                                 benchmark_version=args.suite)