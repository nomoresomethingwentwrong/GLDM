import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from math import ceil
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=int, default=1, help='model number')

args = parser.parse_args()
model_num = args.model
model = ['vae', 'aae', 'wae', 'biaae'][model_num]

if model == 'biaae':
    mol_file = 'cond_generation_res/' + model + '_generated_molecules_and_sa_scores.pkl'
else:
    mol_file = 'cond_generation_res/ldm_con_' + model + '_generated_molecules_and_sa_scores.pkl'
with open(mol_file, 'rb') as f:
    mols = pickle.load(f)

mols_org = defaultdict(lambda: [])
for k, v in mols.items():
    ref_smi = k.split('_')[0]
#     if ref_smi == 'Nc1ccccc1NC(=O)c1ccc(CNc2nccc(-c3cccnc3)n2)cc1':
#         print(k, v)
    mols_org[ref_smi] += v['generated_mols']

meta_data = pd.read_csv('protein/metadata.csv', header=0)
ref_smis = meta_data['smiles'].to_list()
ref_smis = list(set(ref_smis))
opts = StereoEnumerationOptions(tryEmbedding=True, unique=True)

for smi in ref_smis:
    if smi not in mols_org.keys():
        print(smi)
        raise ValueError
    else:
        generated_mols = mols_org[smi]
        np.random.seed(2023)
        np.random.shuffle(generated_mols)
        os.makedirs(os.path.join('ligand', model, smi), exist_ok=True)
        counter = 0
        for m in tqdm(generated_mols):
            if counter >= 100:
                break
            name = Chem.MolToSmiles(m)
            isomers = tuple(EnumerateStereoisomers(m, options=opts))
            if len(isomers) == 0:
                continue
            isomers = [Chem.AddHs(i) for i in isomers]
            for i in isomers:
                AllChem.EmbedMolecule(i, randomSeed=2023)
                with Chem.SDWriter(os.path.join('ligand/'+model, smi, name+'.sdf')) as w:
                    w.write(i)
            counter += 1