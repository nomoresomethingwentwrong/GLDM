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

'''
with open("/data/ongh0068/l1000/l1000_biaae/protein_target_to_l1000_smiles.pkl", 'rb') as f:
    meta_data = pickle.load(f)
inv_meta_data = {}
for k,v in meta_data.items():
    for x in v:
        inv_meta_data.setdefault(x, []).append(k)

aae_file = '../ldm/cond_generation_res/ldm_con_' + model + '_generated_molecules_and_sa_scores.pkl'
with open(aae_file, 'rb') as f:
    aae_mols = pickle.load(f)

aae_org = defaultdict(lambda: [])
for k, v in aae_mols.items():
    ref_smi = k.split('_')[0]
#     if ref_smi == 'Nc1ccccc1NC(=O)c1ccc(CNc2nccc(-c3cccnc3)n2)cc1':
#         print(k, v)
    aae_org[ref_smi] += v['generated_mols']

proteins = ['AKT1', 'AKT2', 'AURKB', 'EGFR', 'PIK3CA', 'SMAD3', 'HDAC1', 'TP53', 'MTOR']
log_path = 'logs/original/'

# find ligand with best binding affinity for each protein
ref_dict = {}

for pr in proteins:
    ref_dict[pr] = {}
    best = 0.0
    best_smi = ''
    for file_name in os.listdir(log_path+pr):
        with open(os.path.join(log_path + pr, file_name), 'r') as f:
            lines = f.readlines()
#         score = float(lines[19][11:17])
        lines = lines[16:]
        n = ceil(len(lines)/14)
        arr = np.zeros(n)
        for i in range(n):
            arr[i] = float(lines[i*14+3][11:17])
        score = np.min(arr)
        smi = file_name.split('.')[0]
        if score < best and smi in aae_org.keys():
            best = score
            best_smi = smi
    print('protein ', pr, 'best score ', best)
    ref_dict[pr]['score'] = best
    ref_dict[pr]['smiles'] = best_smi

selected = {v['smiles']:k for k, v in ref_dict.items()}
print(selected)

# Save generated mols in sdf format
opts = StereoEnumerationOptions(tryEmbedding=True, unique=True)

for ref_smi, pr in selected.items():
#     if pr in ['AKT1', 'EGFR', 'TP53', 'PIK3CA']:
#         continue
    print(pr)
    mols = aae_org[ref_smi]
    np.random.seed(2023)
    np.random.shuffle(mols)
    selected_mols = mols
#     selected_mols = np.random.choice(mols, 100)
    print(len(selected_mols))
    os.makedirs(os.path.join('ligand', model, ref_smi), exist_ok=True)

    counter = 0
    for m in tqdm(selected_mols):
        if counter >= 100:
            break
        name = Chem.MolToSmiles(m)
        isomers = tuple(EnumerateStereoisomers(m, options=opts))    # some may fail to embed??
        if len(isomers) == 0:
            continue
        isomers = [Chem.AddHs(i) for i in isomers]
        for i in isomers:
            AllChem.EmbedMolecule(i, randomSeed=2023)
            with Chem.SDWriter(os.path.join('ligand/'+model, ref_smi, name+'.sdf')) as w:
                w.write(i)  
        
        counter += 1
'''

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