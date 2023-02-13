import sys
import itertools
from pytorch_lightning import LightningModule
from model_utils import GenericMLP, MoLeROutput, PropertyRegressionMLP
from encoder import GraphEncoder, PartialGraphEncoder
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from decoder import MLPDecoder
import torch
import numpy as np
from utils import BIG_NUMBER, pprint_pyg_obj
from decoding_utils import (
    construct_decoder_states,
    sample_indices_from_logprobs,
    batch_decoder_states,
)
from torchvision import transforms


sys.path.append("../moler_reference")
from molecule_generation.utils.training_utils import get_class_balancing_weights

from molecule_generation.utils.moler_decoding_utils import (
    restrict_to_beam_size_per_mol,
    MoLeRDecoderState,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationEdgeCandidateInfo,
)

class AbstractModel(LightningModule):
    """Common decoding methods for each model (decoding at inference time doesn't change)"""
    def __init__(self):
        super(AbstractModel, self).__init__()

    def _is_atom_type(self, node_type):
        if not self.uses_motifs:
            return True
        else:
            return node_type in self._atom_types

    def _add_atom_or_motif(
        self,
        decoder_state,
        node_type,
        logprob,
        choice_info,
    ):
        # If we are running with motifs, we need to check whether `node_type` is an atom or a motif.
        if self._is_atom_type(node_type):
            return (
                MoLeRDecoderState.new_with_added_atom(
                    decoder_state,
                    node_type,
                    atom_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                False,
            )
        else:
            return (
                MoLeRDecoderState.new_with_added_motif(
                    decoder_state,
                    node_type,
                    motif_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                True,
            )

    @property
    def uses_motifs(self):
        return self._uses_motifs

    @property
    def uses_categorical_features(self):
        return self._node_categorical_num_classes is not None

    @property
    def full_graph_encoder(self):
        return self._full_graph_encoder

    @property
    def partial_graph_encoder(self):
        return self._partial_graph_encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def mean_log_var_mlp(self):
        return self._mean_log_var_mlp

    @property
    def latent_dim(self):
        return self._latent_repr_dim

    def _decoder_pick_first_atom_types(
        self,
        decoder_states,
        sampling_mode="greedy",
        num_samples=1,
    ):
        with torch.no_grad():
            # We only need the molecule representations.
            latent_representations = torch.stack(
                [state.molecule_representation for state in decoder_states]
            )

            first_node_type_logits = self.decoder.pick_first_node_type(
                latent_representations=latent_representations
            )  # Shape [G, NT + 1]

            first_atom_type_logprobs = torch.nn.functional.log_softmax(
                first_node_type_logits[:, 1:], # because index 0 corresponds to UNK
                dim=1,  
            )  # Shape [G, NT]

            first_atom_type_pick_results = []

            # Iterate over each of the rows independently, sampling for each input state:
            for state_first_atom_type_logprobs in first_atom_type_logprobs:
                picked_atom_type_indices = sample_indices_from_logprobs(
                    num_samples, sampling_mode, state_first_atom_type_logprobs
                )

                this_state_results = []

                for picked_atom_type_idx in picked_atom_type_indices:
                    pick_logprob = state_first_atom_type_logprobs[picked_atom_type_idx]
                    picked_atom_type_idx += (
                        1  # Revert the stripping out of the UNK (index 0) type
                    )

                    this_state_results.append(
                        (
                            self._index_to_node_type_map[picked_atom_type_idx.item()],
                            pick_logprob,
                        )
                    )

                first_atom_type_pick_results.append(
                    (this_state_results, state_first_atom_type_logprobs)
                )
            return first_atom_type_pick_results

    def _pick_attachment_points_for_batch(
        self,
        batch,
        decoder_states,
        num_samples=1,
        sampling_mode="greedy",
    ):

        initial_focus_atom_idx = batch.candidate_attachment_points_ptr[:-1]

        initial_focus_atoms = batch.candidate_attachment_points[initial_focus_atom_idx]

        candidate_attachment_points = batch.candidate_attachment_points

        graph_representations, node_representations = self.partial_graph_encoder(
            node_features=batch.x,
            partial_graph_node_categorical_features=batch.node_categorical_features,
            edge_index=batch.edge_index,
            edge_features=batch.partial_graph_edge_features,
            # Here we choose an arbitrary attachment point as a focus atom; this does not matter
            # since later all candidate attachment points are marked with the in-focus bit.
            graph_to_focus_node_map=initial_focus_atoms,
            candidate_attachment_points=candidate_attachment_points,
            batch_index=batch.batch,
        )

        attachment_point_selection_logits = self.decoder.pick_attachment_point(
            input_molecule_representations=batch.latent_representation,
            partial_graph_representations=graph_representations,
            node_representations=node_representations,
            node_to_graph_map=batch.batch,
            candidate_attachment_points=candidate_attachment_points,
        )  # Shape: [CA]

        attachment_point_to_graph_map = batch.batch[candidate_attachment_points]

        # TODO(krmaziar): Consider tensorizing the code below. For that, we need some equivalent of
        # `unsorted_segment_argmax`.
        logits_by_graph = [[] for _ in range(len(decoder_states))]

        for logit, graph_id in zip(
            attachment_point_selection_logits, attachment_point_to_graph_map
        ):
            logits_by_graph[graph_id].append(logit)
        attachment_point_pick_results = []
        for old_decoder_state, attachment_point_logits in zip(
            decoder_states, logits_by_graph
        ):

            attachment_point_logprobs = torch.nn.functional.log_softmax(
                torch.tensor(attachment_point_logits), dim=0
            )#.numpy()
            picked_att_point_indices = sample_indices_from_logprobs(
                num_samples, sampling_mode, attachment_point_logprobs
            )

            this_state_results = []
            for attachment_point_pick_idx in picked_att_point_indices:
                attachment_point_pick = old_decoder_state.candidate_attachment_points[
                    attachment_point_pick_idx
                ]
                attachment_point_logprob = attachment_point_logprobs[
                    attachment_point_pick_idx
                ]

                this_state_results.append(
                    (attachment_point_pick, attachment_point_logprob)
                )
            attachment_point_pick_results.append(this_state_results)

        return attachment_point_pick_results, logits_by_graph

    def _decoder_pick_attachment_points(
        self, decoder_states, sampling_mode="greedy", num_samples=1
    ):
        if len(decoder_states) == 0:
            return [], np.zeros(shape=(0,))

        def add_state_to_attachment_point_choice_batch(
            decoder_state_features, decoder_state
        ):
            decoder_state_features[
                "candidate_attachment_points"
            ] = decoder_state.candidate_attachment_points
            return decoder_state_features

        attachment_point_pick_results = []
        logits_by_graph = []

        for batch, decoder_states_batch in batch_decoder_states(
            decoder_states=decoder_states,
            batch_size=16,
            atom_featurisers=self._atom_featurisers,
            motif_vocabulary=self._motif_vocabulary,
            add_state_to_batch_callback=add_state_to_attachment_point_choice_batch,
        ):
            # print('_decoder_pick_attachment_points')
            # pprint_pyg_obj(batch, True)
            with torch.no_grad():
                (
                    pick_results_for_batch,
                    logits_for_batch,
                ) = self._pick_attachment_points_for_batch(
                    batch=batch,
                    decoder_states=decoder_states_batch,
                    num_samples=num_samples,
                    sampling_mode=sampling_mode,
                )
                attachment_point_pick_results.extend(pick_results_for_batch)
                logits_by_graph.extend(logits_for_batch)

        return attachment_point_pick_results, logits_by_graph

    def _pick_edges_for_batch(
        self,
        batch,
        decoder_states,
        num_samples=1,
        sampling_mode="greedy",
        store_generation_traces=False,
    ):
        with torch.no_grad():
            # print('batch.focus_atoms,', batch.focus_atoms)
            graph_representations, node_representations = self._partial_graph_encoder(
                node_features=batch.x,
                partial_graph_node_categorical_features=batch.node_categorical_features,
                edge_index=batch.edge_index,
                edge_features=batch.partial_graph_edge_features,
                # Here we choose an arbitrary attachment point as a focus atom; this does not matter
                # since later all candidate attachment points are marked with the in-focus bit.
                graph_to_focus_node_map=batch.focus_atoms,
                candidate_attachment_points=torch.zeros(
                    size=(0,), device=self.full_graph_encoder._dummy_param.device
                ),
                batch_index=batch.batch,
            )

            batch_candidate_edge_targets = batch.candidate_edge_targets
            batch_candidate_edge_type_masks = batch.candidate_edge_type_masks

            edge_candidate_logits, edge_type_logits = self.decoder.pick_edge(
                input_molecule_representations=batch.latent_representation,
                partial_graph_representations=graph_representations,
                node_representations=node_representations,
                num_graphs_in_batch=len(batch.ptr) - 1,
                focus_node_idx_in_batch=batch.focus_atoms,
                node_to_graph_map=batch.batch,
                candidate_edge_targets=batch_candidate_edge_targets.long(),
                candidate_edge_features=batch.candidate_edge_features.float(),
            )

            # We now need to unpack the results, which is a bit fiddly because the "no more edges"
            # logits are bunched together at the end for all input graphs...
            num_total_edge_candidates = torch.sum(
                batch.decoder_state_to_num_candidate_edges
            )
            edge_candidate_offset = 0
            picked_edges = []

            for state_idx, (
                decoder_state,
                decoder_state_num_edge_candidates,
            ) in enumerate(
                zip(decoder_states, batch.decoder_state_to_num_candidate_edges)
            ):
                # We had no valid candidates -> Easy out:
                if decoder_state_num_edge_candidates == 0:
                    picked_edges.append(([], None))
                    continue

                # Find the edge targets for this decoder state, in the original node index:
                edge_targets = batch_candidate_edge_targets[
                    edge_candidate_offset : edge_candidate_offset
                    + decoder_state_num_edge_candidates
                ]
                edge_targets_orig_idx = edge_targets - batch.ptr[state_idx]

                # Get logits for edge candidates for this decoder state:
                decoder_state_edge_candidate_logits = edge_candidate_logits[
                    edge_candidate_offset : edge_candidate_offset
                    + decoder_state_num_edge_candidates
                ]
                decoder_state_no_edge_logit = edge_candidate_logits[
                    num_total_edge_candidates + state_idx
                ]

                decoder_state_edge_cand_logprobs = torch.nn.functional.log_softmax(
                    torch.cat(
                        [
                            decoder_state_edge_candidate_logits,
                            torch.tensor([decoder_state_no_edge_logit], device = self.full_graph_encoder._dummy_param.device),
                        ]
                    ),
                    dim=0,
                )

                # Before we continue, generate the information for the trace visualisation:
                molecule_generation_edge_choice_info = None
                if store_generation_traces:
                    # Set up the edge candidate info
                    candidate_edge_type_logits = edge_type_logits[
                        edge_candidate_offset : edge_candidate_offset
                        + decoder_state_num_edge_candidates
                    ]
                    candidate_edge_type_mask = batch_candidate_edge_type_masks[
                        edge_candidate_offset : edge_candidate_offset
                        + decoder_state_num_edge_candidates
                    ]
                    masked_candidate_edge_type_logits = (
                        candidate_edge_type_logits
                        - BIG_NUMBER * (1 - candidate_edge_type_mask)
                    )

                    # Loop over the edge candidates themselves.
                    molecule_generation_edge_candidate_info = []
                    for edge_idx, (target, score, logprob) in enumerate(
                        zip(
                            edge_targets_orig_idx,
                            decoder_state_edge_candidate_logits,
                            decoder_state_edge_cand_logprobs,
                        )
                    ):
                        molecule_generation_edge_candidate_info.append(
                            MoleculeGenerationEdgeCandidateInfo(
                                target_node_idx=target,
                                score=score,
                                logprob=logprob,
                                correct=None,
                                type_idx_to_logprobs=torch.nn.functional.log_softmax(
                                    masked_candidate_edge_type_logits[edge_idx, :]
                                ),
                            )
                        )
                    molecule_generation_edge_choice_info = MoleculeGenerationEdgeChoiceInfo(
                        focus_node_idx=decoder_state.focus_atom,
                        partial_molecule_adjacency_lists=decoder_state.adjacency_lists,
                        candidate_edge_infos=molecule_generation_edge_candidate_info,
                        no_edge_score=decoder_state_no_edge_logit,
                        no_edge_logprob=decoder_state_edge_cand_logprobs[-1],
                        no_edge_correct=None,
                    )

                # Collect (sampling) results for this state:
                this_state_results = []
                picked_edge_cand_indices = sample_indices_from_logprobs(
                    num_samples, sampling_mode, decoder_state_edge_cand_logprobs
                )
                for picked_edge_cand_idx in picked_edge_cand_indices:
                    picked_cand_logprob = decoder_state_edge_cand_logprobs[
                        picked_edge_cand_idx
                    ]
                    # Handle case of having no edge is better:
                    if (
                        picked_edge_cand_idx
                        == len(decoder_state_edge_cand_logprobs) - 1
                    ):
                        this_state_results.append((None, picked_cand_logprob))
                    else:
                        # Otherwise, we need to find the target of that edge, in the original
                        # (unbatched) node index:
                        picked_edge_partner = edge_targets_orig_idx[
                            picked_edge_cand_idx
                        ]

                        # Next, identify the edge type for this choice:
                        edge_type_mask = batch_candidate_edge_type_masks[
                            edge_candidate_offset + picked_edge_cand_idx
                        ]
                        cand_edge_type_logprobs = torch.nn.functional.log_softmax(
                            edge_type_logits[
                                edge_candidate_offset + picked_edge_cand_idx
                            ]
                            - BIG_NUMBER * (1 - edge_type_mask),
                            dim=0,
                        )
                        picked_edge_types = sample_indices_from_logprobs(
                            num_samples, sampling_mode, cand_edge_type_logprobs
                        )
                        for picked_edge_type in picked_edge_types:
                            picked_edge_logprob = (
                                picked_cand_logprob
                                + cand_edge_type_logprobs[picked_edge_type]
                            )
                            this_state_results.append(
                                (
                                    (picked_edge_partner, picked_edge_type),
                                    picked_edge_logprob,
                                )
                            )
                picked_edges.append(
                    (this_state_results, molecule_generation_edge_choice_info)
                )
                edge_candidate_offset += decoder_state_num_edge_candidates
            return picked_edges

    def _decoder_pick_new_bond_types(
        self,
        decoder_states,
        sampling_mode="greedy",
        store_generation_traces=False,
        num_samples=1,
    ):
        def add_state_to_edge_batch(decoder_state_features, decoder_state):
            #     TODO add all these into MolerData to get the right offset.

            decoder_state_features["focus_atoms"] = decoder_state.focus_atom
            (
                candidate_targets,
                candidate_bond_type_mask,
            ) = decoder_state.get_bond_candidate_targets()
            num_edge_candidates = len(candidate_targets)
            decoder_state_features["candidate_edge_targets"] = candidate_targets
            # the following will just be batch.ptr
            #         decoder_state_features["candidate_edge_targets_offset"] = batch["nodes_in_batch"]
            decoder_state_features[
                "candidate_edge_type_masks"
            ] = candidate_bond_type_mask
            decoder_state_features[
                "candidate_edge_features"
            ] = decoder_state.compute_bond_candidate_features(candidate_targets)
            decoder_state_features[
                "decoder_state_to_num_candidate_edges"
            ] = num_edge_candidates
            return decoder_state_features

        batch_generator = batch_decoder_states(
            decoder_states=decoder_states,
            batch_size=16,
            atom_featurisers=self._atom_featurisers,
            motif_vocabulary=self._motif_vocabulary,
            add_state_to_batch_callback=add_state_to_edge_batch,
        )
        # print('_decoder_pick_new_bond_types')
        # for b, d in batch_generator:
        #     pprint_pyg_obj(b, verbose = True)
        # pprint_pyg_obj(next(iter(batch_generator)))

        picked_edges_generator = (
            self._pick_edges_for_batch(
                b, d, num_samples, sampling_mode, store_generation_traces
            )
            for b, d in batch_generator
        )
        return itertools.chain.from_iterable(picked_edges_generator)

    def _decoder_pick_new_atom_types(
        self,
        decoder_states,
        sampling_mode="greedy",
        num_samples=1,
    ):
        def add_state_to_atom_choice_batch(decoder_state_features, decoder_state):
            #     TODO add all these into MolerData to get the right offset.
            decoder_state_features["prior_focus_atoms"] = decoder_state.prior_focus_atom
            return decoder_state_features

        batch_generator = batch_decoder_states(
            decoder_states=decoder_states,
            batch_size=16,
            atom_featurisers=self._atom_featurisers,
            motif_vocabulary=self._motif_vocabulary,
            add_state_to_batch_callback=add_state_to_atom_choice_batch,
        )
        atom_type_pick_generator = (
            self._pick_new_atom_types_for_batch(batch, num_samples, sampling_mode)
            for batch, _ in batch_generator
        )
        return itertools.chain.from_iterable(atom_type_pick_generator)

    def _pick_new_atom_types_for_batch(
        self, batch, num_samples=1, sampling_mode="greedy"
    ):
        # print('batch.prior_focus_atoms')
        # pprint_pyg_obj(batch, True)
        with torch.no_grad():
            graph_representations, _ = self.partial_graph_encoder(
                partial_graph_node_categorical_features=batch.node_categorical_features,
                node_features=batch.x,
                edge_index=batch.edge_index.long(),
                edge_features=batch.partial_graph_edge_features,
                # Note: This whole prior_focus_atom is a bit of a hack. During training, we use the
                # same graph for predict-no-more-bonds and predict-next-atom-type. Hence, during
                # training, we always have at least one in-focus node per graph, and not
                # matching that would be confusing to the model. Hence, we simulate this behaviour:
                graph_to_focus_node_map=batch.prior_focus_atoms,
                candidate_attachment_points=torch.zeros(
                    size=(0,), device=self.full_graph_encoder._dummy_param.device
                ),
                batch_index=batch.batch,
            )
            
            node_type_logits = self.decoder.pick_node_type(
                input_molecule_representations=batch.latent_representation,
                graph_representations=graph_representations,
                graphs_requiring_node_choices=torch.arange(0, len(batch.ptr) - 1),
            )  # Shape [G, NT + 1]

            # Remove the first column, corresponding to UNK, which we never want to produce, but add it
            # back later so that the type lookup indices work out:
            atom_type_logprobs = torch.nn.functional.log_softmax(
                node_type_logits[:, 1:], dim=1
            )#.numpy()  # Shape [G, NT]

            atom_type_pick_results = []
            # Iterate over each of the rows independently, sampling for each input state:
            for state_atom_type_logprobs in atom_type_logprobs:
                picked_atom_type_indices = sample_indices_from_logprobs(
                    num_samples, sampling_mode, state_atom_type_logprobs
                )

                this_state_results = []
                for picked_atom_type_idx in picked_atom_type_indices:
                    pick_logprob = state_atom_type_logprobs[picked_atom_type_idx]
                    picked_atom_type_idx += (
                        1  # Revert the stripping out of the UNK (index 0) type
                    )
                    # This is the case in which we picked the "no further nodes" virtual node type:
                    if picked_atom_type_idx >= self._num_node_types:
                        this_state_results.append((None, pick_logprob))
                    else:
                        picked_atom_type = self._index_to_node_type_map[
                            picked_atom_type_idx.item()
                        ]
                        this_state_results.append((picked_atom_type, pick_logprob))
                atom_type_pick_results.append(
                    (this_state_results, state_atom_type_logprobs)
                )
            return atom_type_pick_results

    def decode(
        self,
        latent_representations,
        initial_molecules=None,
        mol_ids=None,
        store_generation_traces=False,
        max_num_steps=120,
        beam_size=1,
        sampling_mode="greedy",
    ):
        # use this for initialising decoder states when using initial scaffolds
        decoder_states_empty, decoder_states_non_empty = construct_decoder_states(
            motif_vocabulary=self._motif_vocabulary,
            latent_representations=latent_representations,
            uses_motifs=self._uses_motifs,
            initial_molecules=initial_molecules,
            mol_ids=mol_ids,
            store_generation_traces=store_generation_traces,
        )

        # Step 0: Pick first node types for states that do not have an initial molecule.
        first_node_pick_results = self._decoder_pick_first_atom_types(
            decoder_states=decoder_states_empty,
            num_samples=beam_size,
            sampling_mode=sampling_mode,
        )

        decoder_states = decoder_states_non_empty

        for decoder_state, (first_node_type_picks, first_node_type_logprobs) in zip(
            decoder_states_empty, first_node_pick_results
        ):
            for first_node_type_pick, first_node_type_logprob in first_node_type_picks:
                # Set up generation trace storing variables, populating if needed.
                atom_choice_info = None
                if store_generation_traces:
                    atom_choice_info = MoleculeGenerationAtomChoiceInfo(
                        node_idx=0,
                        true_type_idx=None,
                        type_idx_to_prob=np.exp(first_node_type_logprobs),
                    )

                new_decoder_state, added_motif = self._add_atom_or_motif(
                    decoder_state,
                    first_node_type_pick,
                    logprob=first_node_type_logprob,
                    choice_info=atom_choice_info,
                )

                last_atom_id = new_decoder_state.molecule.GetNumAtoms() - 1

                if added_motif:
                    # To make all asserts happy, pretend we chose an attachment point.
                    new_decoder_state._focus_atom = last_atom_id

                # Mark all initial nodes as visited.
                new_decoder_state = MoLeRDecoderState.new_with_focus_marked_as_visited(
                    old_state=new_decoder_state, focus_node_finished_logprob=0.0
                )

                # Set the prior focus atom similarly to the start-from-scaffold case.
                new_decoder_state._prior_focus_atom = last_atom_id

                decoder_states.append(new_decoder_state)

        num_steps = 0
        while num_steps < max_num_steps:
            # This will hold the results after this decoding step, grouped by input mol id:
            new_decoder_states = []
            num_steps += 1
            # Step 1: Split decoder states into subsets, dependent on what they need next:
            (
                require_atom_states,
                require_bond_states,
                require_attachment_point_states,
            ) = ([], [], [])
            for decoder_state in decoder_states:
                # No focus atom => needs a new atom
                if decoder_state.focus_atom is None:
                    require_atom_states.append(decoder_state)
                # Focus atom has invalid index => decoding finished, just push forward unchanged:
                elif decoder_state.focus_atom < 0:
                    new_decoder_states.append(decoder_state)
                else:
                    require_bond_states.append(decoder_state)

            # Check if we are done:
            if (len(require_atom_states) + len(require_bond_states)) == 0:
                # print("I: Decoding finished")
                break

            # Step 2: For states that require a new atom, try to pick one:
            node_pick_results = self._decoder_pick_new_atom_types(
                decoder_states=require_atom_states,
                num_samples=beam_size,
                sampling_mode=sampling_mode,
            )

            for decoder_state, (node_type_picks, node_type_logprobs) in zip(
                require_atom_states, node_pick_results
            ):
                for node_type_pick, node_type_logprob in node_type_picks:
                    # Set up generation trace storing variables, populating if needed.
                    atom_choice_info = None
                    if store_generation_traces:
                        atom_choice_info = MoleculeGenerationAtomChoiceInfo(
                            node_idx=decoder_state.prior_focus_atom + 1,
                            true_type_idx=None,
                            type_idx_to_prob=np.exp(node_type_logprobs),
                        )

                    # If the decoder says we need no new atoms anymore, we are finished. Otherwise,
                    # start adding more bonds:
                    if node_type_pick is None:
                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Finished decoding - p={node_type_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_for_finished_decoding(
                                old_state=decoder_state,
                                finish_logprob=node_type_logprob,
                                atom_choice_info=atom_choice_info,
                            )
                        )
                    else:
                        new_decoder_state, added_motif = self._add_atom_or_motif(
                            decoder_state,
                            node_type_pick,
                            logprob=node_type_logprob,
                            choice_info=atom_choice_info,
                        )

                        if added_motif:
                            require_attachment_point_states.append(new_decoder_state)
                        else:
                            require_bond_states.append(new_decoder_state)
            
            if self.uses_motifs:
                # Step 2': For states that require picking an attachment point, pick one:
                require_attachment_point_states = restrict_to_beam_size_per_mol(
                    require_attachment_point_states, beam_size
                )
                
                (
                    attachment_pick_results,
                    attachment_pick_logits,
                ) = self._decoder_pick_attachment_points(
                    decoder_states=require_attachment_point_states,
                    sampling_mode=sampling_mode,
                )
                # print('attachment_pick_results, attachment_pick_logits', attachment_pick_results, attachment_pick_logits)
                for (
                    decoder_state,
                    attachment_point_picks,
                    attachment_point_logits,
                ) in zip(
                    require_attachment_point_states,
                    attachment_pick_results,
                    attachment_pick_logits,
                ):
                    for (
                        attachment_point_pick,
                        attachment_point_logprob,
                    ) in attachment_point_picks:
                        attachment_point_choice_info = None

                        if store_generation_traces:
                            attachment_point_choice_info = MoleculeGenerationAttachmentPointChoiceInfo(
                                partial_molecule_adjacency_lists=decoder_state.adjacency_lists,
                                motif_nodes=decoder_state.atoms_to_mark_as_visited,
                                candidate_attachment_points=decoder_state.candidate_attachment_points,
                                candidate_idx_to_prob=torch.nn.functional.log_softmax(
                                    attachment_point_logits, dim=-1
                                ),
                                correct_attachment_point_idx=None,
                            )

                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Picked attachment point {attachment_point_pick} - p={attachment_point_logprob:5f}")
                        require_bond_states.append(
                            MoLeRDecoderState.new_with_focus_on_attachment_point(
                                decoder_state,
                                attachment_point_pick,
                                focus_atom_logprob=attachment_point_logprob,
                                attachment_point_choice_info=attachment_point_choice_info,
                            )
                        )
            else:
                assert not require_attachment_point_states

            # Step 3: Pick fresh bonds and populate the next round of decoding steps:
            require_bond_states = restrict_to_beam_size_per_mol(
                require_bond_states, beam_size
            )
            bond_pick_results = self._decoder_pick_new_bond_types(
                decoder_states=require_bond_states,
                store_generation_traces=store_generation_traces,
                sampling_mode=sampling_mode,
            )
            for (decoder_state, (bond_picks, edge_choice_info)) in zip(
                require_bond_states, bond_pick_results
            ):
                if len(bond_picks) == 0:
                    # There were no valid options for this bonds, so we treat this as if
                    # predicting no more bonds with probability 1.0:
                    # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: No more allowed bonds to node {decoder_state.focus_atom}")
                    new_decoder_states.append(
                        MoLeRDecoderState.new_with_focus_marked_as_visited(
                            decoder_state,
                            focus_node_finished_logprob=0,
                            edge_choice_info=edge_choice_info,
                        )
                    )
                    continue

                for (bond_pick, bond_pick_logprob) in bond_picks:
                    # If the decoder says we need no more bonds for the current focus node,
                    # we mark this and put the decoder state back for the next expansion round:
                    if bond_pick is None:
                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Finished connecting bonds to node {decoder_state.focus_atom} - p={bond_pick_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_with_focus_marked_as_visited(
                                decoder_state,
                                focus_node_finished_logprob=bond_pick_logprob,
                                edge_choice_info=edge_choice_info,
                            )
                        )
                    else:
                        (picked_bond_target, picked_bond_type) = bond_pick

                        # print(I {decoder_state.molecule_id} {decoder_state.logprob:12f}: Adding {decoder_state.focus_atom}-{picked_bond_type}->{picked_bond_target} - p={bond_pick_logprob:5f}")
                        new_decoder_states.append(
                            MoLeRDecoderState.new_with_added_bond(
                                old_state=decoder_state,
                                target_atom_idx=int(
                                    picked_bond_target
                                ),  # Go from np.int32 to pyInt
                                bond_type_idx=picked_bond_type.item(),
                                bond_logprob=bond_pick_logprob,
                                edge_choice_info=edge_choice_info,
                            )
                        )

            # Everything is done, restrict to the beam width, and go back to the loop start:
            decoder_states = restrict_to_beam_size_per_mol(
                new_decoder_states, beam_size
            )

        return decoder_states

    def validation_epoch_end(self, outputs):
        # decoder 50 random molecules using fixed random seed
        if self._decode_on_validation_end:
            if self.current_epoch < 3:
                pass
            else:
                generator = torch.Generator(device = self.full_graph_encoder._dummy_param.device).manual_seed(0)
                latent_vectors = torch.randn(size = (50, 512), generator = generator, device = self.full_graph_encoder._dummy_param.device)
                decoder_states = self.decode(latent_representations = latent_vectors)
                print([Chem.MolToSmiles(decoder_states[i].molecule) for i in range(len(decoder_states))])
                try:
                    pil_imgs = [Draw.MolToImage(decoder_states[i].molecule) for i in range(len(decoder_states))]
                    pil_img_tensors = [transforms.ToTensor()(pil_img).permute(1,2,0) for pil_img in pil_imgs]
                    
                    for pil_img_tensor in pil_img_tensors:
                        self.logger.experiment.add_image('sample_molecules', pil_img_tensor, self.current_epoch)
                except Exception as e:
                    print(e)

class BaseModel(AbstractModel):
    def __init__(self, params, dataset, num_train_batches=1, batch_size=1):
        """Params is a nested dictionary with the relevant parameters."""
        super(BaseModel, self).__init__()
        self._init_params(params, dataset)
        self.save_hyperparameters()
        if "training_hyperparams" in params:
            self._training_hyperparams = params["training_hyperparams"]
        else:
            self._training_hyperparams = None
        self._params = params
        self._num_train_batches = num_train_batches
        self._batch_size = batch_size
        self._use_oclr_scheduler = params["use_oclr_scheduler"]
        self._decode_on_validation_end = params['decode_on_validation_end']
        self._using_cyclical_anneal = params['using_cyclical_anneal']
        # Graph encoders
        self._full_graph_encoder = GraphEncoder(**self._params["full_graph_encoder"])
        self._partial_graph_encoder = PartialGraphEncoder(
            **self._params["partial_graph_encoder"]
        )

        # Replace this with any other latent space mapping techniques eg diffusion
        self._mean_log_var_mlp = GenericMLP(**self._params["mean_log_var_mlp"])

        # MLP for regression task on graph properties
        self._include_property_regressors = "graph_properties" in self._params
        if self._include_property_regressors:
            self._graph_property_pred_loss_weight = self._params[
                "graph_property_pred_loss_weight"
            ]
            self._property_predictors = torch.nn.ModuleDict()
            for prop_name, prop_params in self._params["graph_properties"].items():
                prop_stddev = dataset.metadata.get(f"{prop_name}_stddev")
                if not (prop_params.get("normalise_loss", True)):
                    prop_stddev = None
                self._property_predictors[prop_name] = PropertyRegressionMLP(
                    **prop_params["mlp"],
                    property_stddev=prop_stddev,
                )

        # MLP decoders
        self._decoder = MLPDecoder(self._params["decoder"])

        # params for latent space
        self._latent_sample_strategy = self._params["latent_sample_strategy"]
        self._latent_repr_dim = self._params["latent_repr_size"]
        self._kl_divergence_weight = self._params["kl_divergence_weight"]
        self._kl_divergence_annealing_beta = self._params[
            "kl_divergence_annealing_beta"
        ]

    def _init_params(self, params, dataset):
        """
        Initialise class weights for next node prediction and placefolder for
        motif/node embeddings.
        """

        self._motif_vocabulary = dataset.metadata.get("motif_vocabulary")
        self._uses_motifs = self._motif_vocabulary is not None

        self._node_categorical_num_classes = len(dataset.node_type_index_to_string)

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in params:
                self._node_categorical_features_embedding = None

        if self.uses_motifs:
            # Record the set of atom types, which will be a subset of all node types.
            self._atom_types = set(
                dataset._atom_type_featuriser.index_to_atom_type_map.values()
            )

        self._index_to_node_type_map = dataset.node_type_index_to_string
        self._atom_featurisers = dataset._metadata["feature_extractors"]
        self._num_node_types = dataset.num_node_types


    def sample_from_latent_repr(self, latent_repr):
        mean_and_log_var = self.mean_log_var_mlp(latent_repr)
        # mean_and_log_var = torch.clamp(mean_and_log_var, min=-10, max=10)
        # perturb latent repr
        mu = mean_and_log_var[:, : self.latent_dim]  # Shape: [V, MD]
        log_var = mean_and_log_var[:, self.latent_dim :]  # Shape: [V, MD]

        # result_representations: shape [num_partial_graphs, latent_repr_dim]
        z = self.reparametrize(mu, log_var)
        # p, q, z = self.reparametrize(mu, log_var)

        return mu, log_var, z
        # return p, q, z

    def reparametrize(self, mu, log_var):
        """Samples a different noise vector for each partial graph.
        TODO: look into the other sampling strategies."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        # return p, q, z

    def forward(self, batch):
        moler_output = self._run_step(batch)
        return (
            moler_output.first_node_type_logits,
            moler_output.node_type_logits,
            moler_output.edge_candidate_logits,
            moler_output.edge_type_logits,
            moler_output.attachment_point_selection_logits,
        )

    def _run_step(self, batch):
        # Obtain graph level representation of original molecular graph
        input_molecule_representations = self.full_graph_encoder(
            original_graph_node_categorical_features=batch.original_graph_node_categorical_features,
            node_features=batch.original_graph_x.float(),
            edge_index=batch.original_graph_edge_index,
            edge_features=batch.original_graph_edge_features,  # can be edge_type or edge_attr
            batch_index=batch.original_graph_x_batch,
        )

        # Obtain graph level representation of the partial graph
        partial_graph_representions, node_representations = self.partial_graph_encoder(
            partial_graph_node_categorical_features=batch.partial_node_categorical_features,
            node_features=batch.x,
            edge_index=batch.edge_index.long(),
            edge_features=batch.partial_graph_edge_features,
            graph_to_focus_node_map=batch.focus_node,
            candidate_attachment_points=batch.valid_attachment_point_choices,
            batch_index=batch.batch,
        )

        # Apply latent sampling strategy
        # mu, log_var, latent_representation = self.sample_from_latent_repr(
        #     input_molecule_representations
        # )
        mu, log_var, latent_representation = self.sample_from_latent_repr(
            input_molecule_representations
        )

        # Forward pass through decoder
        (
            first_node_type_logits,
            node_type_logits,
            edge_candidate_logits,
            edge_type_logits,
            attachment_point_selection_logits,
        ) = self.decoder(
            input_molecule_representations=latent_representation,
            graph_representations=partial_graph_representions,
            graphs_requiring_node_choices=batch.correct_node_type_choices_batch.unique(),
            # edge selection
            node_representations=node_representations,
            num_graphs_in_batch=len(batch.ptr) - 1,
            focus_node_idx_in_batch=batch.focus_node,
            node_to_graph_map=batch.batch,
            candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
            candidate_edge_features=batch.edge_features,
            # attachment selection
            candidate_attachment_points=batch.valid_attachment_point_choices.long(),
        )

        # NOTE: loss computation will be done in lightning module
        return MoLeROutput(
            first_node_type_logits=first_node_type_logits,
            node_type_logits=node_type_logits,
            edge_candidate_logits=edge_candidate_logits,
            edge_type_logits=edge_type_logits,
            attachment_point_selection_logits=attachment_point_selection_logits,
            mu=mu,
            log_var=log_var,
            # p=p,
            # q=q,
            latent_representation=latent_representation,
        )

    def compute_loss(self, moler_output, batch):
        # num_correct_node_type_choices = (
        #     batch.correct_node_type_choices_ptr.unique().shape[-1] - 1
        # )
        node_type_multihot_labels = batch.correct_node_type_choices  # .view(
        #     num_correct_node_type_choices, -1
        # )

        first_node_type_multihot_labels = (
            batch.correct_first_node_type_choices
        )  # .view(len(batch.ptr) -1, -1)

        loss = self.decoder.compute_decoder_loss(
            # node selection
            node_type_logits=moler_output.node_type_logits,
            node_type_multihot_labels=node_type_multihot_labels,
            # first node selection
            first_node_type_logits=moler_output.first_node_type_logits,
            first_node_type_multihot_labels=first_node_type_multihot_labels,
            # edge selection
            num_graphs_in_batch=len(batch.ptr) - 1,
            node_to_graph_map=batch.batch,
            candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
            edge_candidate_logits=moler_output.edge_candidate_logits,  # as is
            per_graph_num_correct_edge_choices=batch.num_correct_edge_choices,
            edge_candidate_correctness_labels=batch.correct_edge_choices,
            no_edge_selected_labels=batch.stop_node_label,
            # edge type selection
            correct_edge_choices=batch.correct_edge_choices,
            valid_edge_types=batch.valid_edge_types,
            edge_type_logits=moler_output.edge_type_logits,
            edge_type_onehot_labels=batch.correct_edge_types,
            # attachement point
            attachment_point_selection_logits=moler_output.attachment_point_selection_logits,
            attachment_point_candidate_to_graph_map=batch.valid_attachment_point_choices_batch.long(),
            attachment_point_correct_choices=batch.correct_attachment_point_choice.long(),
        )

        return loss

    def compute_property_prediction_loss(self, latent_representation, batch):
        """TODO: Since graph property regression is more of a auxillary loss than anything, this function will be
        decoupled in the future into `compute_properties` and `compute_property_prediction_loss` so that
        it can be passed into the `_run_step` function and returned in MolerOutput."""
        property_prediction_losses = {}
        for prop_name, property_predictor in self._property_predictors.items():
            predictions = property_predictor(latent_representation)
            property_prediction_losses[prop_name] = property_predictor.compute_loss(
                predictions=predictions, labels=batch[prop_name]
            )
        # sum up all the property prediction losses
        return sum([loss for loss in property_prediction_losses.values()])

    def step(self, batch):
        moler_output = self._run_step(batch)

        loss_metrics = {}
        loss_metrics['decoder_loss'] = self.compute_loss(moler_output=moler_output, batch=batch)
        if self._include_property_regressors:
            loss_metrics['property_prediction_loss'] = (
                self._graph_property_pred_loss_weight
                * self.compute_property_prediction_loss(
                    latent_representation=moler_output.latent_representation,
                    batch=batch,
                )
            )
        # print("log_var", torch.max(moler_output.log_var))
        kld_summand = torch.square(moler_output.mu)
        + torch.exp(moler_output.log_var)
        - moler_output.log_var
        - 1
        loss_metrics['kld_loss'] = torch.mean( kld_summand)/2.0
        # loss_metrics['kld_loss'] = torch.distributions.kl_divergence(
        #     moler_output.q, moler_output.p
        # ).mean()
        # kld weight will start from 0 and increase to the original amount.

        annealing_factor = self.trainer.global_step % (self._num_train_batches // 4) if self._using_cyclical_anneal else self.trainer.global_step

        loss_metrics['kld_weight'] = (
            (  # cyclical anealing where each cycle will span 1/4 of the training epoch
                1.0
                - self._kl_divergence_annealing_beta
                ** annealing_factor
            )
            * self._kl_divergence_weight
        )

        loss_metrics['kld_loss'] *= loss_metrics['kld_weight']

        loss_metrics['loss'] = sum(loss_metrics.values())

        logs = loss_metrics
        return loss_metrics['loss'], logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        for metric in logs:
            self.log(f"train_{metric}", logs[metric], batch_size=self._batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        for metric in logs:
            self.log(f"val_{metric}", logs[metric], batch_size=self._batch_size)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._training_hyperparams["max_lr"]
        )

        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self._training_hyperparams["max_lr"],
        #     betas=(0.9, 0.999),
        # )
        if self._use_oclr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self._training_hyperparams["max_lr"],
                div_factor=self._training_hyperparams["div_factor"],
                three_phase=self._training_hyperparams["three_phase"],
                epochs=self.trainer.max_epochs,
                # number of times step() is called by the scheduler per epoch
                # take the number of batches // frequency of calling the scheduler
                steps_per_epoch=self._num_train_batches // self.trainer.max_epochs,
            )

            lr_scheduler_params = {}
            lr_scheduler_params["scheduler"] = lr_scheduler

            lr_scheduler_params["interval"] = "step"
            frequency_of_lr_scheduler_step = self.trainer.max_epochs
            lr_scheduler_params[
                "frequency"
            ] = frequency_of_lr_scheduler_step  # number of batches to wait before calling lr_scheduler.step()

            optimizer_dict = {}
            optimizer_dict["optimizer"] = optimizer
            optimizer_dict["lr_scheduler"] = lr_scheduler_params
            return optimizer_dict
        else:
            return optimizer

