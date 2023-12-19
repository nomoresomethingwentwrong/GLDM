import pandas as pd
import numpy as np
import subprocess

def download(id):
    subprocess.run(['wget', 'https://files.rcsb.org/download/' + id + '.pdb', '-O', 'protein/original_files/' + id + '.pdb'])

def split_lig_pr(pdb_id, lig_id):
    # subprocess.run(['grep', 'ATOM', 'protein/original_files/' + pdb_id + '.pdb', r'>', 'protein/' + pdb_id + '_' + 'protein.pdb'])
    # subprocess.run(['grep', lig_id, 'protein/original_files/' + pdb_id + '.pdb', r'>', 'pocket/' + pdb_id + '_' + 'pocket.pdb'])
    subprocess.run([f'grep ATOM protein/original_files/{pdb_id}.pdb > protein/{pdb_id}_protein.pdb'], shell=True, capture_output=True)
    subprocess.run([f'grep {lig_id} protein/original_files/{pdb_id}.pdb > pocket/{pdb_id}_pocket.pdb'], shell=True, capture_output=True)

metadata = pd.read_csv('protein/metadata.csv', header=0)
for index, row in metadata.iterrows():
    pdb_id = row['pdb']
    lig_id = row['ligand']
    download(pdb_id)
    split_lig_pr(pdb_id, lig_id)