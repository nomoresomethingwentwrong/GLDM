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

'''
with open("/data/ongh0068/l1000/l1000_biaae/protein_target_to_l1000_smiles.pkl", 'rb') as f:
    meta_data = pickle.load(f)

inv_meta_data = {}
for k,v in meta_data.items():
    for x in v:
        inv_meta_data.setdefault(x, []).append(k)
        
models = ['original']
proteins = ['AKT1', 'AKT2', 'AURKB', 'EGFR', 'PIK3CA', 'SMAD3', 'HDAC1', 'TP53', 'MTOR']
for model in models:
    for pr in proteins:
        log_path = os.path.join('logs/' + model, pr)
        out_path = os.path.join('poses/' + model, pr)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)

for model in models:
    for ref_smi, proteins in tqdm(inv_meta_data.items()):
        # escape_smi = re.sub(r'([()])', r'\\\1', ref_smi)
        escape_smi = ref_smi
        for pr in proteins:
            protein_file = 'protein/' + pr + '.pdb'
            ligand_file = os.path.join('ligand/' + model, escape_smi + '.sdf')
            log_file = os.path.join('logs/' + model, pr, escape_smi + '.log')
            out_file = os.path.join('poses/' + model, pr, escape_smi + '.sdf.gz')
            subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', protein_file, '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '-q'], stdout=subprocess.DEVNULL)
'''

meta_data = pd.read_csv('protein/metadata.csv', header=0)
meta_data = meta_data.iloc[15:18, :]
print(meta_data)
for index, row in meta_data.iterrows():
    protein = row['protein']
    smiles = row['smiles']
    pdb_id = row['pdb']
    lig_id = row['ligand']
    protein_file = 'protein/' + pdb_id + '_protein.pdb'
    ligand_file = 'ligand/original/' + smiles + '.sdf'
    autobox_ligand_file = 'pocket/' + pdb_id + '_pocket.pdb'
    log_file = 'logs/original/' + pdb_id + '.log'
    out_file = 'poses/original/' + pdb_id + '.sdf'
    subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', autobox_ligand_file, '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '-q'], stdout=subprocess.DEVNULL)