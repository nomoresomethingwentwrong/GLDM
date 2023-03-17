# for each row, we want to generate 1000 molecules
# but for each row, we have a variable number of choices for the control gene exp
# as well as a variable number of choices for the tumour gene expression
# so we compute the number of possible pairs => then we divide 1000 by it
# then during inference, we condition on the difference of each pair


# Compute all possible gene expression difference vectors
import itertools  
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.Fraggle.FraggleSim import GetFraggleSimilarity
import numpy as np
import torch 
from tqdm import tqdm

def generate_similar_molecules_with_gene_exp_diff(
    control_idx, 
    tumour_idx,
    dataset,
    model,
    num_samples = 1000,
    device = 'cuda:0'
):
    possible_pairs = np.array(list(itertools.product(control_idx, tumour_idx)))

    control_idx_batched = possible_pairs[:, 0]
    tumour_idx_batched = possible_pairs[:, 1]

    control_gene_exp_batched = dataset._gene_exp_controls[control_idx_batched]
    tumour_gene_exp_batched = dataset._gene_exp_tumour[tumour_idx_batched]
    difference_gene_exp_batched = tumour_gene_exp_batched - control_gene_exp_batched

    # Create num_samples//num_diff_vectors random vectors 
    if num_samples > difference_gene_exp_batched.shape[0]:
        num_rand_vectors_required = num_samples//difference_gene_exp_batched.shape[0]
        random_vectors = torch.randn(num_rand_vectors_required, 512, device = device)
        # repeat each gene expression difference vector in its place a number of times
        # equal to the number of random vectors using repeat_interleave
        # then repeat the random vectors batchwise so that we can align the random vectors 
        # with the gene expression differences 
        # Eg given 114 gene expression diff vectors, we will have 8 random vectors
        # then for each gene expresison vector, we want to match it with each of the 
        # 8 random vectors individually
        difference_gene_exp_batched = torch.tensor(difference_gene_exp_batched, device = device)
        difference_gene_exp_batched = torch.repeat_interleave(difference_gene_exp_batched, num_rand_vectors_required, dim = 0)
        random_vectors = random_vectors.repeat(possible_pairs.shape[0], 1)
    
    else:
        num_rand_vectors_required = num_samples
        # since number of samples is less than the number of gene expressions
        # we need to truncate the gene expressions too
        difference_gene_exp_batched = torch.tensor(difference_gene_exp_batched[:num_samples, :], device = device)
        random_vectors = torch.randn(num_rand_vectors_required, 512, device = device)


    conditioned_random_vectors = model.condition_on_gene_expression(
        latent_representation=random_vectors,
        gene_expressions=difference_gene_exp_batched,
    )

    # compute similarity score between all 1000 generated molecules and the actual molecule
    # take the max similarity score
    decoder_states = model.decode(latent_representations =conditioned_random_vectors, max_num_steps = 120)
    molecules = [decoder_state.molecule for decoder_state in decoder_states]

    return molecules

def compute_max_similarity(
    candidate_molecules, 
    reference_smile, 
    radius = 3,
    nBits = 1024
):
    m_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits) for mol in candidate_molecules]
    maccs_fps = [MACCSkeys.GenMACCSKeys(mol) for mol in candidate_molecules]
    reference_mol = Chem.MolFromSmiles(reference_smile)
    reference_smile_m_fp =  AllChem.GetMorganFingerprintAsBitVect(reference_mol,radius=radius, nBits=nBits) 
    reference_smile_maccs_fp = MACCSkeys.GenMACCSKeys(reference_mol)


    m_fp_tanimoto_sim = DataStructs.BulkTanimotoSimilarity(reference_smile_m_fp, m_fps)
    maccs_fp_tanimoto_sim = DataStructs.BulkTanimotoSimilarity(reference_smile_maccs_fp, maccs_fps)
    fraggle_sims = [sim for sim, match in [GetFraggleSimilarity(reference_mol,candidate_mol) for candidate_mol in candidate_molecules]]
    return {
        'max_morgan_fp_tanimoto_sim': max(m_fp_tanimoto_sim),
        'max_maccs_fp_tanimoto_sim': max(maccs_fp_tanimoto_sim),
        'max_fraggle_sim': max(fraggle_sims)
    }


def internal_diversity(
    generated_molecules,
    radius = 3,
    nBits = 1024,
):
    m_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits) for mol in generated_molecules]
    tanimoto_sim_sum = 0
    for fp in tqdm(m_fps):
        tanimoto_sim_sum += sum(DataStructs.BulkTanimotoSimilarity(fp, [other_fp for other_fp in m_fps if other_fp != fp]))
    return 1- 1/(len(generated_molecules)*(len(generated_molecules)-1))*tanimoto_sim_sum