import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=int, default=1, help='model number')

args = parser.parse_args()
model_num = args.model
model = ['vae', 'aae', 'wae'][model_num]

# Load generated mols
filename = '../ldm/cond_generation_res/ldm_con_' + model + '_generated_molecules_and_sa_scores.pkl'
with open(filename, 'rb') as f:
    data_ori = pickle.load(f)

data = defaultdict(lambda: [])
for k, v in data_ori.items():
    ref_smi = k.split('_')[0]
    data[ref_smi] += v['generated_mols']

# new_keys = list(data.keys())[146:]
# data = {k: data[k] for k in new_keys}
# Save generated mols in sdf format
for ref_smi, mols in tqdm(data.items()):
    with Chem.SDWriter(os.path.join('ligand/'+model, ref_smi+'.sdf')) as w:
      for m in mols:
        Chem.SanitizeMol(m)
        m = Chem.AddHs(m)
        try: 
            AllChem.EmbedMolecule(m,randomSeed=2023)
        except Exception as e:
            print(e)
            # continue
        w.write(m)