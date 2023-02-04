import torch
from utils import (
    safe_divide_loss,
    compute_neglogprob_for_multihot_objective,
    traced_unsorted_segment_log_softmax,
)
from model_utils import GenericMLP
import torch.nn.functional as F

distance_truncation = 10
BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7

# def softmax_cross_entropy_with_logits(logits, targets, reduce="none"):
#     """Pytorch analogue for tf.nn.softmax_cross_entropy_with_logits."""
#     # https://stackoverflow.com/questions/46218566/pytorch-equivalence-for-softmax-cross-entropy-with-logits
#     if reduce == "none":
#         return torch.nn.functional.cross_entropy(logits, targets, reduction = reduce)



class MLPDecoder(torch.nn.Module):
    """Returns graph level representation of the molecules."""

    def __init__(
        self,
        params,  # nested dictionary of parameters for each MLP
    ):
        super(MLPDecoder, self).__init__()
        self._dummy_param = torch.nn.Parameter(
            torch.empty(0)
        )  # for inferrring device of model

        # TODO include each loss weight for weighted loss computation in the loss

        # First Node Selection
        self._first_node_type_selector = GenericMLP(
            **params["first_node_type_selector"]
        )

        # Node selection
        self._node_type_selector = GenericMLP(**params["node_type_selector"])
        self._node_type_loss_weights = None
        if params['use_node_type_loss_weights']: # False by default
            self._node_type_loss_weights = params[
                "node_type_loss_weights"
            ]  # cannot move to gpu yet because trainer has not been instantiated

        # Edge selection
        self._no_more_edges_representation = torch.nn.Parameter(
            torch.empty(*params["no_more_edges_repr"]), requires_grad=True
        )
        torch.nn.init.kaiming_normal_(
            self._no_more_edges_representation,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        self._edge_candidate_scorer = GenericMLP(**params["edge_candidate_scorer"])
        self._edge_type_selector = GenericMLP(**params["edge_type_selector"])
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        # since we want to truncate the distance, we should have an embedding layer for it
        self._distance_embedding_layer = torch.nn.Embedding(distance_truncation, 1)

        # Attachment Point Selection
        self._attachment_point_selector = GenericMLP(
            **params["attachment_point_selector"]
        )

    def pick_node_type(
        self,
        input_molecule_representations,
        graph_representations,
        graphs_requiring_node_choices,
    ):

        relevant_graph_representations = input_molecule_representations[
            graphs_requiring_node_choices
        ]
        relevant_input_molecule_representations = graph_representations[
            graphs_requiring_node_choices
        ]
        original_and_calculated_graph_representations = torch.cat(
            (relevant_graph_representations, relevant_input_molecule_representations),
            axis=-1,
        )
        node_type_logits = self._node_type_selector(
            original_and_calculated_graph_representations
        )
        return node_type_logits

    def compute_node_type_selection_loss(
        self, node_type_logits, node_type_multihot_labels
    ):
        per_node_decision_logprobs = torch.nn.functional.log_softmax(
            node_type_logits, dim=-1
        )
        # Shape: [NTP, NT + 1]

        # number of correct choices for each of the partial graphs that require node choices
        per_node_decision_num_correct_choices = torch.sum(
            node_type_multihot_labels, keepdim=True, axis=-1
        )
        # Shape [NTP, 1]

        per_correct_node_decision_normalised_neglogprob = (
            compute_neglogprob_for_multihot_objective(
                logprobs=per_node_decision_logprobs[
                    :, :-1
                ],  # separate out the no node prediction
                multihot_labels=node_type_multihot_labels,
                per_decision_num_correct_choices=per_node_decision_num_correct_choices,
            )
        )  # Shape [NTP, NT]

        no_node_decision_correct = (
            per_node_decision_num_correct_choices == 0.0
        )  # Shape [NTP, 1]
        per_correct_no_node_decision_neglogprob = -(
            per_node_decision_logprobs[:, -1]
            * torch.squeeze(no_node_decision_correct).float()
        )  # Shape [NTP]

        if self._node_type_loss_weights is not None:
            per_correct_node_decision_normalised_neglogprob *= (
                self._node_type_loss_weights[:-1].to(self._dummy_param.device)
            )
            per_correct_no_node_decision_neglogprob *= self._node_type_loss_weights[
                -1
            ].to(self._dummy_param.device)

        # Loss is the sum of the masked (no) node decisions, averaged over number of decisions made:
        total_node_type_loss = torch.sum(
            per_correct_node_decision_normalised_neglogprob
        ) + torch.sum(per_correct_no_node_decision_neglogprob)
        node_type_loss = safe_divide_loss(
            total_node_type_loss, node_type_multihot_labels.shape[0]
        )

        return node_type_loss

    def pick_first_node_type(self, latent_representations):
        return self._first_node_type_selector(latent_representations)

    def compute_first_node_type_selection_loss(
        self,
        first_node_type_logits,
        first_node_type_multihot_labels,
    ):
        per_graph_logprobs = torch.nn.functional.log_softmax(
            first_node_type_logits, dim=-1
        )
        per_graph_num_correct_choices = torch.sum(
            first_node_type_multihot_labels, axis=-1, keepdims=True
        )
        per_graph_normalised_neglogprob = compute_neglogprob_for_multihot_objective(
            logprobs=per_graph_logprobs,
            multihot_labels=first_node_type_multihot_labels,
            per_decision_num_correct_choices=per_graph_num_correct_choices,
        )
        if self._node_type_loss_weights is not None:
            per_graph_normalised_neglogprob *= self._node_type_loss_weights[:-1].to(
                self._dummy_param.device
            )

        first_node_type_loss = safe_divide_loss(
            torch.sum(per_graph_normalised_neglogprob),
            first_node_type_multihot_labels.shape[0],
        )
        return first_node_type_loss

    def pick_edge(
        self,
        input_molecule_representations,
        partial_graph_representations,
        node_representations,
        num_graphs_in_batch,  # len(batch.ptr) - 1
        focus_node_idx_in_batch,  # batch.focus_node
        node_to_graph_map,  # batch.batch
        candidate_edge_targets,  # batch.valid_edge_choices[:, 1]
        candidate_edge_features,  # batch.edge_features
    ):
        focus_node_representations = node_representations[focus_node_idx_in_batch]

        graph_and_focus_node_representations = torch.cat(
            (
                input_molecule_representations,
                partial_graph_representations,
                focus_node_representations,
            ),
            axis=-1,
        )

        # Explanation: at each step, there is a focus node, which is the node we are
        # focusing on right now in terms of adding another edge to it. When adding a new
        # edge, the edge can be between the focus node and a variety of other nodes.
        # This is likely based on valency, and in reality, it is possible that none of the
        # edge choices are correct (when that generation step is a node addition step)
        # and not an edge addition step. Regardless, we still want to consider the candidates
        # "target" refers to the node at the other end of the candidate edge
        valid_target_to_graph_map = node_to_graph_map[candidate_edge_targets]
        graph_and_focus_node_representations_per_edge_candidate = (
            graph_and_focus_node_representations[valid_target_to_graph_map]
        )
        edge_candidate_target_node_representations = node_representations[
            candidate_edge_targets
        ]

        # The zeroth element of edge_features is the graph distance. We need to look that up
        # in the distance embeddings:
        truncated_distances = candidate_edge_features[:, 0].minimum(
            (
                torch.ones(
                    len(candidate_edge_features), device=self._dummy_param.device
                )
                * (distance_truncation - 1)
            )
        )
        # shape: [CE]

        distance_embedding = self._distance_embedding_layer(truncated_distances.long())

        # Concatenate all the node features, to form focus_node -> target_node edge features
        edge_candidate_representation = torch.cat(
            (
                graph_and_focus_node_representations_per_edge_candidate,
                edge_candidate_target_node_representations,
                distance_embedding,
                candidate_edge_features[:, 1:],
            ),
            axis=-1,
        )

        stop_edge_selection_representation = torch.cat(
            [
                graph_and_focus_node_representations,
                torch.tile(
                    self._no_more_edges_representation,
                    dims=(num_graphs_in_batch, 1),
                ),
            ],
            axis=-1,
        )  # shape: [PG, MD + PD + 2 * VD*(num_layers+1) + FD]

        edge_candidate_and_stop_features = torch.cat(
            [edge_candidate_representation, stop_edge_selection_representation], axis=0
        )  # shape: [CE + PG, MD + PD + 2 * VD*(num_layers+1) + FD]
        edge_candidate_logits = torch.squeeze(
            self._edge_candidate_scorer(edge_candidate_and_stop_features),
            axis=-1,
        )  # shape: [CE + PG]
        edge_type_logits = self._edge_type_selector(
            edge_candidate_representation
        )  # shape: [CE, ET]

        return edge_candidate_logits, edge_type_logits

    def compute_edge_candidate_selection_loss(
        self,
        num_graphs_in_batch,  # len(batch.ptr)-1
        node_to_graph_map,  # batch.batch
        candidate_edge_targets,  # batch_features["valid_edge_choices"][:, 1]
        edge_candidate_logits,  # as is
        per_graph_num_correct_edge_choices,  # batch.num_correct_edge_choices
        edge_candidate_correctness_labels,  # correct edge choices
        no_edge_selected_labels,  # stop node label
    ):

        # First, we construct full labels for all edge decisions, which are the concat of
        # edge candidate logits and the logits for choosing no edge:
        edge_correctness_labels = torch.cat(
            [edge_candidate_correctness_labels, no_edge_selected_labels.float()],
            axis=0,
        )  # Shape: [CE + PG]

        # To compute a softmax over all candidate edges (and the "no edge" choice) corresponding
        # to the same graph, we first need to build the map from each logit to the corresponding
        # graph id. Then, we can do an unsorted_segment_softmax using that map:
        edge_candidate_to_graph_map = node_to_graph_map[candidate_edge_targets]
        # add the end bond labels to the end
        edge_candidate_to_graph_map = torch.cat(
            (
                edge_candidate_to_graph_map,
                torch.arange(0, num_graphs_in_batch, device=self._dummy_param.device),
            )
        )

        edge_candidate_logprobs = traced_unsorted_segment_log_softmax(
            logits=edge_candidate_logits,
            segment_ids=edge_candidate_to_graph_map,
        )  # Shape: [CE + PG]

        # Compute the edge loss with the multihot objective.
        # For a single graph with three valid choices (+ stop node) of which two are correct,
        # we may have the following:
        #  edge_candidate_logprobs = log([0.05, 0.5, 0.4, 0.05])
        #  per_graph_num_correct_edge_choices = [2]
        #  edge_candidate_correctness_labels = [0.0, 1.0, 1.0]
        #  edge_correctness_labels = [0.0, 1.0, 1.0, 0.0]
        # To get the loss, we simply look at the things in edge_candidate_logprobs that correspond
        # to correct entries.
        # However, to account for the _multi_hot nature, we scale up each entry of
        # edge_candidate_logprobs by the number of correct choices, i.e., consider the
        # correct entries of
        #  log([0.05, 0.5, 0.4, 0.05]) + log([2, 2, 2, 2]) = log([0.1, 1.0, 0.8, 0.1])
        # In this form, we want to have each correct entry to be as near possible to 1.
        # Finally, we normalise loss contributions to by-graph, by dividing the crossentropy
        # loss by the number of correct choices (i.e., in the example above, this results in
        # a loss of -((log(1.0) + log(0.8)) / 2) = 0.11...).

        # Note: per_graph_num_correct_edge_choices does not include the choice of an edge to
        # the stop node, so can be zero.
        per_graph_num_correct_edge_choices = torch.maximum(
            per_graph_num_correct_edge_choices,
            torch.ones(
                per_graph_num_correct_edge_choices.shape,
                device=self._dummy_param.device,
            ),
        )  # Shape: [PG]
        per_edge_candidate_num_correct_choices = torch.index_select(per_graph_num_correct_edge_choices, 0, edge_candidate_to_graph_map)
        # per_graph_num_correct_edge_choices[
        #     edge_candidate_to_graph_map
        # ]
        # Shape: [CE]
        per_correct_edge_neglogprob = -(
            (
                edge_candidate_logprobs
                + torch.log(per_edge_candidate_num_correct_choices +SMALL_NUMBER)
            )
            * edge_correctness_labels
            / per_edge_candidate_num_correct_choices
        )  # Shape: [CE]

        # Normalise by number of graphs for which we made edge selection decisions:
        edge_loss = safe_divide_loss(
            torch.sum(per_correct_edge_neglogprob), num_graphs_in_batch
        )

        return edge_loss

    def compute_edge_type_selection_loss(
        self,
        valid_edge_types,  # batch.valid_edge_types
        edge_type_logits,
        correct_edge_choices,  # batch.correct_edge_choices
        edge_type_onehot_labels,  # batch.correct_edge_types
    ):
        correct_target_indices = correct_edge_choices != 0
        edge_type_logits_for_correct_edges = edge_type_logits[correct_target_indices]

        # The `valid_edge_types` tensor is equal to 1 when the edge is valid (it may be invalid due
        # to valency constraints), 0 otherwise.
        # We want to multiply the selection probabilities by this mask. Because the logits are in
        # log space, we instead subtract a large value from the logits wherever this mask is zero.
        scaled_edge_mask = (
            1 - valid_edge_types.float()
        ) * BIG_NUMBER  # Shape: [CCE, ET]
        masked_edge_type_logits = (
            edge_type_logits_for_correct_edges - scaled_edge_mask
        )  # Shape: [CCE, ET]
        edge_type_loss = torch.nn.functional.cross_entropy(
            masked_edge_type_logits, edge_type_onehot_labels.float(), reduction = 'none'
        )
        # Normalise by the number of edges for which we needed to pick a type:
        # instead of mean, we must use safe divide because the batch can have zero edges
        # requring edge types
        edge_type_loss = safe_divide_loss(
            torch.sum(edge_type_loss), len(edge_type_loss)
        )

        return edge_type_loss

    def pick_attachment_point(
        self,
        input_molecule_representations,  # as is
        partial_graph_representations,  # partial_graph_representations
        node_representations,  # as is
        node_to_graph_map,  # batch.batch
        candidate_attachment_points,  # valid_attachment_point_choices
    ):
        original_and_calculated_graph_representations = torch.cat(
            [input_molecule_representations, partial_graph_representations],
            axis=-1,
        )  # Shape: [PG, MD + PD]

        # Map attachment point candidates to their respective partial graphs.
        partial_graphs_for_attachment_point_choices = node_to_graph_map[
            candidate_attachment_points
        ]  # Shape: [CA]

        # To score an attachment point, we condition on the representations of input and partial
        # graphs, along with the representation of the attachment point candidate in question.
        attachment_point_representations = torch.cat(
            [
                original_and_calculated_graph_representations[
                    partial_graphs_for_attachment_point_choices
                ],
                node_representations[candidate_attachment_points],
            ],
            axis=-1,
        )  # Shape: [CA, MD + PD + VD*(num_layers+1)]

        attachment_point_selection_logits = torch.squeeze(
            self._attachment_point_selector(attachment_point_representations), axis=-1
        )

        return attachment_point_selection_logits

    def compute_attachment_point_selection_loss(
        self,
        attachment_point_selection_logits,  # as is
        attachment_point_candidate_to_graph_map,  # = batch2.valid_attachment_point_choices_batch.long(),
        attachment_point_correct_choices,  # = batch2.correct_attachment_point_choices
    ):
        # Compute log softmax of the logits within each partial graph.
        attachment_point_candidate_logprobs = (
            traced_unsorted_segment_log_softmax(
                logits=attachment_point_selection_logits,
                segment_ids=attachment_point_candidate_to_graph_map,
            )
            * 1.0
        )  # Shape: [CA]
        attachment_point_correct_choice_neglogprobs = (
            -torch.index_select(attachment_point_candidate_logprobs, 0, attachment_point_correct_choices)
            # -attachment_point_candidate_logprobs[attachment_point_correct_choices]
        )

        # Shape: [AP]

        attachment_point_selection_loss = safe_divide_loss(
            torch.sum(attachment_point_correct_choice_neglogprobs),
            attachment_point_correct_choice_neglogprobs.shape[0],
        )
        return attachment_point_selection_loss

    def compute_decoder_loss(
        self,
        # node selection
        node_type_logits,
        node_type_multihot_labels,
        # first node selection
        first_node_type_logits,
        first_node_type_multihot_labels,
        # edge selection
        num_graphs_in_batch,
        node_to_graph_map,
        candidate_edge_targets,
        edge_candidate_logits,
        per_graph_num_correct_edge_choices,
        edge_candidate_correctness_labels,
        no_edge_selected_labels,
        # edge type selection
        correct_edge_choices,
        valid_edge_types,
        edge_type_logits,
        edge_type_onehot_labels,
        # attachement point
        attachment_point_selection_logits,  # as is
        attachment_point_candidate_to_graph_map,  # = batch2.valid_attachment_point_choices_batch.long(),
        attachment_point_correct_choices,
    ):
        # Compute node selection loss
        node_selection_loss = self.compute_node_type_selection_loss(
            node_type_logits, node_type_multihot_labels
        )

        # Compute first node selection loss
        first_node_selection_loss = self.compute_first_node_type_selection_loss(
            first_node_type_logits,
            first_node_type_multihot_labels,
        )

        # Compute edge selection loss
        edge_loss = self.compute_edge_candidate_selection_loss(
            num_graphs_in_batch,
            node_to_graph_map,
            candidate_edge_targets,
            edge_candidate_logits,
            per_graph_num_correct_edge_choices,
            edge_candidate_correctness_labels,
            no_edge_selected_labels,
        )

        edge_type_loss = self.compute_edge_type_selection_loss(
            valid_edge_types,
            edge_type_logits,
            correct_edge_choices,
            edge_type_onehot_labels,
        )
        # Compute attachement point selection loss
        attachment_point_loss = self.compute_attachment_point_selection_loss(
            attachment_point_selection_logits,  # as is
            attachment_point_candidate_to_graph_map,  # = batch2.valid_attachment_point_choices_batch.long(),
            attachment_point_correct_choices,
        )

        # TODO Weighted sum of the losses and return it for backpropagation in
        # the lightning module
        # TODO add weights of losses into params

        return (
            node_selection_loss
            + 0.07 * first_node_selection_loss
            + edge_loss
            + edge_type_loss
            + attachment_point_loss
        )

    def forward(
        self,
        input_molecule_representations,  # latent representation
        graph_representations,
        graphs_requiring_node_choices,
        # edge selection
        node_representations,
        num_graphs_in_batch,
        focus_node_idx_in_batch,
        node_to_graph_map,
        candidate_edge_targets,
        candidate_edge_features,
        # attachment selection
        candidate_attachment_points,
    ):
        # Compute node logits
        node_logits = self.pick_node_type(
            input_molecule_representations,
            graph_representations,
            graphs_requiring_node_choices,
        )

        # Compute first node logits
        first_node_logits = self.pick_first_node_type(input_molecule_representations)

        # Compute edge logits
        edge_candidate_logits, edge_type_logits = self.pick_edge(
            input_molecule_representations,
            graph_representations,
            node_representations,
            num_graphs_in_batch,
            focus_node_idx_in_batch,
            node_to_graph_map,
            candidate_edge_targets,
            candidate_edge_features,
        )
        # Compute attachment point logits
        attachment_point_selection_logits = self.pick_attachment_point(
            input_molecule_representations,  # latent representation
            graph_representations,  # partial_graph_representations
            node_representations,  # as is
            node_to_graph_map,
            candidate_attachment_points,
        )
        # return all logits
        return (
            first_node_logits,
            node_logits,
            edge_candidate_logits,
            edge_type_logits,
            attachment_point_selection_logits,
        )
