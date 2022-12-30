
import numpy as np
from rdkit import Chem
import sys
sys.path.append("../moler_reference")

from molecule_generation.chem.molecule_dataset_utils import BOND_DICT
from molecule_generation.chem.motif_utils import (
    find_motifs_from_vocabulary,
)
from molecule_generation.chem.rdkit_helpers import compute_canonical_atom_order, get_atom_symbol

from molecule_generation.utils.moler_decoding_utils import (
    MoLeRDecoderState,
)
from torch_geometric.data import Batch
from dataset import MolerData
import torch
from dataset import EdgeRepresentation

def construct_decoder_states(
    motif_vocabulary, 
    latent_representations,
    uses_motifs,
    initial_molecules,
    mol_ids,
    store_generation_traces,
):
    if initial_molecules is None:
        initial_molecules = [None] * len(latent_representations)

    # Replace `None` in initial_molecules with empty molecules.
    initial_molecules = [
        Chem.Mol() if initMol is None else initMol for initMol in initial_molecules
    ]
    if mol_ids is None:
        mol_ids = range(len(latent_representations))

    decoder_states = []


    # preprocessing for when a scaffold is given
    for graph_repr, init_mol, mol_id in zip(latent_representations, initial_molecules, mol_ids):
        num_free_bond_slots = [0] * len(init_mol.GetAtoms())

        atom_ids_to_remove = []
        atom_ids_to_keep = []

        for atom in init_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                # Atomic number 0 means a placeholder atom that signifies an attachment point.
                bonds = atom.GetBonds()

                if len(bonds) > 1:
                    scaffold = Chem.MolToSmiles(init_mol)
                    raise ValueError(
                        f"Scaffold {scaffold} contains a [*] atom with at least two bonds."
                    )

                if not bonds:
                    # This is a very odd case: either the scaffold we got is disconnected, or
                    # it consists of just a single * atom.
                    scaffold = Chem.MolToSmiles(init_mol)
                    raise ValueError(f"Scaffold {scaffold} contains a [*] atom with no bonds.")

                [bond] = bonds
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()

                neighbour_idx = begin_idx if begin_idx != atom.GetIdx() else end_idx
                num_free_bond_slots[neighbour_idx] += 1

                atom_ids_to_remove.append(atom.GetIdx())
            else:
                atom_ids_to_keep.append(atom.GetIdx())

        if not atom_ids_to_remove:
            # No explicit attachment points, so assume we can connect anywhere.
            num_free_bond_slots = None
        else:
            num_free_bond_slots = [num_free_bond_slots[idx] for idx in atom_ids_to_keep]
            init_mol = Chem.RWMol(init_mol)

            # Remove atoms starting from largest index, so that we don't have to account for
            # indices shifting during removal.
            for atom_idx in reversed(atom_ids_to_remove):
                init_mol.RemoveAtom(atom_idx)

            # Determine how the scaffold atoms will get reordered when we canonicalize it, so we can
            # permute `num_free_bond_slots` appropriately.
            canonical_ordering = compute_canonical_atom_order(init_mol)
            num_free_bond_slots = [num_free_bond_slots[idx] for idx in canonical_ordering]

        # Now canonicalize, which renumbers all the atoms, but we've applied the same
        # renumbering to `num_free_bond_slots` earlier.
        init_mol = Chem.MolFromSmiles(Chem.MolToSmiles(init_mol))

        # Clear aromatic flags in the scaffold, since partial graphs during training never have
        # them set (however we _do_ run `AtomIsAromaticFeatureExtractor`, it just always returns
        # 0 for partial graphs during training).
        # TODO(kmaziarz): Consider fixing this.
        Chem.Kekulize(init_mol, clearAromaticFlags=True)

        init_atom_types = []
        # TODO(kmaziarz): We need to be more careful in how the initial molecule looks like, to
        # make sure that `init_mol`s have correct atom features (e.g. charges).
        for atom in init_mol.GetAtoms():
            init_atom_types.append(get_atom_symbol(atom))
        adjacency_lists = [[] for _ in range(len(BOND_DICT))]
        for bond in init_mol.GetBonds():
            bond_type_idx = BOND_DICT[str(bond.GetBondType())]
            adjacency_lists[bond_type_idx].append(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            )
            adjacency_lists[bond_type_idx].append(
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            )

        if uses_motifs:
            init_mol_motifs = find_motifs_from_vocabulary(
                molecule=init_mol, motif_vocabulary=motif_vocabulary
            )
        else:
            init_mol_motifs = []

        decoder_states.append(
            MoLeRDecoderState(
                molecule_representation=graph_repr,
                molecule_id=mol_id,
                molecule=init_mol,
                atom_types=init_atom_types,
                adjacency_lists=adjacency_lists,
                visited_atoms=[atom.GetIdx() for atom in init_mol.GetAtoms()],
                atoms_to_visit=[],
                focus_atom=None,
                # Pseudo-randomly pick last atom from input:
                prior_focus_atom=len(init_atom_types) - 1,
                generation_steps=[] if store_generation_traces else None,
                motifs=init_mol_motifs,
                num_free_bond_slots=num_free_bond_slots,
            )
        )

    decoder_states_empty = []
    decoder_states_non_empty = []

    for decoder_state in decoder_states:
        if decoder_state.molecule.GetNumAtoms() == 0:
            decoder_states_empty.append(decoder_state)
        else:
            decoder_states_non_empty.append(decoder_state)

    return decoder_states_empty, decoder_states_non_empty

def sample_indices_from_logprobs(
    num_samples, sampling_mode, logprobs
):
    """Samples indices (without replacement) given the log-likelihoods.

    Args:
        num_samples: intended number of samples
        sampling_mode: sampling method (greedy
        logprobs: log-probabilities of selecting appropriate entries

    Returns:
        indices of picked values, shape (n,), where n = min(num_samples, available_samples)

    Note:
        the ordering of returned indices is arbitrary
    """
    num_choices = logprobs.shape[0]
    # indices = np.arange(num_choices)
    num_samples = min(num_samples, num_choices)  # Handle cases where we only have few candidates

    if sampling_mode == 'greedy':
        # Note that this will return the top num_samples indices, but not in order:
        picked_indices = np.argpartition(logprobs, -num_samples)[-num_samples:]
    # elif sampling_mode == DecoderSamplingMode.SAMPLING:
    #     p = np.exp(logprobs)  # Convert to probabilities
    #     # We can only sample values with non-zero probabilities
    #     num_choices = np.sum(p > 0)
    #     num_samples = min(num_samples, num_choices)
    #     picked_indices = np.random.choice(
    #         indices,
    #         size=(num_samples,),
    #         replace=False,
    #         p=p,
    #     )
    else:
        raise ValueError(f"Sampling method {sampling_mode} not known.")

    return picked_indices



def _to_tensor_moler(decoder_state_features, ignore = []):
    for k, v in decoder_state_features.items():
        if k in ignore:
            continue
        decoder_state_features[k] = torch.tensor(decoder_state_features[k])
    return decoder_state_features




        

    
def batch_decoder_states(
    batch_size,
    atom_featurisers, #=dataset._metadata['feature_extractors'] ,
    motif_vocabulary,#=dataset._motif_vocabulary , 
    decoder_states,#=decoder_states,
#     init_batch_callback=init_atom_choice_batch,
    add_state_to_batch_callback,
    type_of_edge_feature = EdgeRepresentation.edge_attr
):
    current_batch = []
    for decoder_state in decoder_states:
        node_features, node_categorical_features = decoder_state.get_node_features(
            atom_featurisers, motif_vocabulary
        )
        # mol_num_nodes = node_features.shape[0]
        
        decoder_state_features = {
            'latent_representation':decoder_state.molecule_representation, 
            'x': node_features,
            'node_categorical_features': node_categorical_features,
        }
        
        edge_indexes = []
        edge_types = []
        for edge_type_idx, adj_list in enumerate(decoder_state.adjacency_lists):
            if len(adj_list) > 0:
                edge_index = np.array(adj_list, dtype=np.int32).T
                edge_indexes += [edge_index]
                """ 
                edge types: 
                single bond => 0
                double bond => 1
                triple bond => 2
                self loop => 3
                """
                edge_types += [edge_type_idx] * len(adj_list)
#         print(edge_indexes)
        decoder_state_features["edge_index"] = (
            np.concatenate(edge_indexes, 1)
            if len(edge_indexes) > 0
            else np.array(edge_indexes)
        )

        if type_of_edge_feature == EdgeRepresentation.edge_type:
            decoder_state_features["partial_graph_edge_features"] = np.array(edge_types)
        elif type_of_edge_feature == EdgeRepresentation.edge_attr:
            edge_attr = edge_types
            decoder_state_features["partial_graph_edge_features"] = np.array(edge_attr)
        
        decoder_state_features = add_state_to_batch_callback(decoder_state_features, decoder_state)
        
        decoder_state_features = _to_tensor_moler(decoder_state_features, ignore = ['latent_representation'])
#         print(decoder_state_features)
        current_batch += [(MolerData(**decoder_state_features), decoder_state)]
        if len(current_batch) == batch_size:
            tmp = current_batch
            current_batch = []
            yield (Batch.from_data_list([i[0] for i in tmp], follow_batch = [
                'correct_edge_choices',
                'correct_edge_types',
                'valid_edge_choices',
                'valid_attachment_point_choices',
                'correct_attachment_point_choice',
                'correct_node_type_choices',
                'original_graph_x',
                'correct_first_node_type_choices',
                # pick attachment points
                'candidate_attachment_points',
                # pick edge
                'candidate_edge_targets'
            ]), [i[1] for i in tmp])
    if len(current_batch) > 0:
        yield (Batch.from_data_list([i[0] for i in current_batch], follow_batch = [
            'correct_edge_choices',
            'correct_edge_types',
            'valid_edge_choices',
            'valid_attachment_point_choices',
            'correct_attachment_point_choice',
            'correct_node_type_choices',
            'original_graph_x',
            'correct_first_node_type_choices',
            # pick attachment points
            'candidate_attachment_points',
            # pick edge
            'candidate_edge_targets'
        ]), [i[1] for i in current_batch])