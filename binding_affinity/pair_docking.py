import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from collections import defaultdict
import re
import subprocess
import argparse

# previous version
# python pair_docking.py -p AKT1 -l O=[N+]\([O-]\)c1ccc\(-c2nc\(-c3ccc\(F\)cc3\)c\(-c3ccncc3\)[nH]2\)cc1
# python pair_docking.py -p AKT2 -l O=C\(O\)c1ccc\(Nc2ncc3c\(n2\)-c2ccc\(Cl\)cc2C\(c2c\(F\)cccc2F\)=NC3\)cc1
# python pair_docking.py -p AURKB -l CC\(Oc1cc\(-c2cnn\(C3CCNCC3\)c2\)cnc1N\)c1c\(Cl\)ccc\(F\)c1Cl
# python pair_docking.py -p EGFR -l CN1CCC\(c2c\(O\)cc\(O\)c3c\(=O\)cc\(-c4ccccc4Cl\)oc23\)C\(O\)C1
# python pair_docking.py -p TP53 -l Cn1cc\(C2=C\(c3cn\(C4CCN\(Cc5ccccn5\)CC4\)c4ccccc34\)C\(=O\)NC2=O\)c2ccccc21
# python pair_docking.py -p PIK3CA -l CNC\(=O\)Nc1ccc\(-c2nc\(N3CC4CCC\(C3\)O4\)c3cnn\(C4CCC5\(CC4\)OCCO5\)c3n2\)cc1
# python pair_docking.py -p MTOR -l Nc1ccc\(-c2ccc3ncc4ccc\(=O\)n\(-c5cccc\(C\(F\)\(F\)F\)c5\)c4c3c2\)cn1
# python pair_docking.py -p SMAD3 -l Cn1c\(=O\)n\(-c2ccc\(C\(C\)\(C\)C#N\)cc2\)c2c3cc\(-c4cnc5ccccc5c4\)ccc3ncc21
# python pair_docking.py -p HDAC1 -l Nc1ccccc1NC\(=O\)c1ccc\(CNC\(=O\)OCc2cccnc2\)cc1

# new version
# python pair_docking.py -p AKT1 -l CN1CCC\(c2c\(O\)cc\(O\)c3c\(=O\)cc\(-c4ccccc4Cl\)oc23\)C\(O\)C1
# python pair_docking.py -p AKT2 -l O=C\(O\)c1ccc\(Nc2ncc3c\(n2\)-c2ccc\(Cl\)cc2C\(c2c\(F\)cccc2F\)=NC3\)cc1
# python pair_docking.py -p AURKB -l CC\(Oc1cc\(-c2cnn\(C3CCNCC3\)c2\)cnc1N\)c1c\(Cl\)ccc\(F\)c1Cl
# python pair_docking.py -p EGFR -l CN1CCC\(c2c\(O\)cc\(O\)c3c\(=O\)cc\(-c4ccccc4Cl\)oc23\)C\(O\)C1
# python pair_docking.py -p TP53 -l Cn1cc\(C2=C\(c3cn\(C4CCN\(Cc5ccccn5\)CC4\)c4ccccc34\)C\(=O\)NC2=O\)c2ccccc21
# python pair_docking.py -p PIK3CA -l CNC\(=O\)Nc1ccc\(-c2nc\(N3CC4CCC\(C3\)O4\)c3cnn\(C4CCC5\(CC4\)OCCO5\)c3n2\)cc1
# python pair_docking.py -p MTOR -l O=c1cc\(N2CCOCC2\)oc2c\(-c3cccc4c3sc3ccccc34\)cccc12
# python pair_docking.py -p SMAD3 -l NS\(=O\)\(=O\)c1cc\(C2\(O\)NC\(=O\)c3ccccc32\)ccc1Cl
# python pair_docking.py -p HDAC1 -l Nc1ccccc1NC\(=O\)c1ccc\(CNC\(=O\)OCc2cccnc2\)cc1
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--protein', type=str, default='AKT1', help='protein name')
parser.add_argument('-l', '--ligand', type=str, default='O=[N+]([O-])c1ccc(-c2nc(-c3ccc(F)cc3)c(-c3ccncc3)[nH]2)cc1', help='ligand smiles')

args = parser.parse_args()
protein = args.protein
ligand = args.ligand

models = ['aae', 'vae', 'wae']

for model in models:
    protein_file = 'protein/' + protein + '.pdb'
    ligand_path = os.path.join('ligand/' + model, ligand)
    log_path = os.path.join('logs/' + model, protein, ligand)
    out_path = os.path.join('poses/' + model, protein, ligand)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    for filename in os.listdir(ligand_path):
        ligand_file = os.path.join(ligand_path, filename)
        if os.path.exists(protein_file) == False or os.path.exists(ligand_file) == False:
            print('protein or ligand file does not exist')
            exit()
        generated_smi = filename.split('.')[0]
        escape_smi = re.sub(r'([()])', r'\\\1', generated_smi)
        log_file = os.path.join(log_path, generated_smi + '.log')
        out_file = os.path.join(out_path, generated_smi + '.sdf')
        subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', protein_file, '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '-q'], stdout=subprocess.DEVNULL)
