import sys

sys.path.append("../moler_reference")
from molecule_generation.utils.training_utils import get_class_balancing_weights
from pytorch_lightning import LightningModule, Trainer, seed_everything
from model_utils import GenericGraphEncoder, GenericMLP, MoLeROutput
from encoder import GraphEncoder

from decoder import MLPDecoder
import torch


class BaseModel(LightningModule):
    def __init__(self, params, dataset):
        """Params is a nested dictionary with the relevant parameters."""
        super(BaseModel, self).__init__()
        self._init_params(params, dataset)
        self._params = params
        # Graph encoders
        self._full_graph_encoder = GraphEncoder(**self._params["full_graph_encoder"])
        self._partial_graph_encoder = GenericGraphEncoder(
            **self._params["partial_graph_encoder"]
        )

        # Replace this with any other latent space mapping techniques eg diffusion
        self._mean_log_var_mlp = GenericMLP(**self._params["mean_log_var_mlp"])

        # MLP decoders
        self._decoder = MLPDecoder(self._params["decoder"])

        # params for latent space
        self._latent_sample_strategy = self._params["latent_sample_strategy"]
        self._latent_repr_dim = self._params["latent_repr_size"]

    def _init_params(self, params, dataset):
        """
        Initialise class weights for next node prediction and placefolder for
        motif/node embeddings.
        """

        # Get some information out from the dataset:
        next_node_type_distribution = dataset.metadata.get(
            "train_next_node_type_distribution"
        )
        class_weight_factor = params.get(
            "node_type_predictor_class_loss_weight_factor", 1.0
        )

        if not (0.0 <= class_weight_factor <= 1.0):
            raise ValueError(
                f"Node class loss weight node_classifier_class_loss_weight_factor must be in [0,1], but is {class_weight_factor}!"
            )
        if class_weight_factor > 0:
            atom_type_nums = [
                next_node_type_distribution[dataset.node_type_index_to_string[type_idx]]
                for type_idx in range(dataset.num_node_types)
            ]
            atom_type_nums.append(next_node_type_distribution["None"])

            self.class_weights = get_class_balancing_weights(
                class_counts=atom_type_nums, class_weight_factor=class_weight_factor
            )
        else:
            self.class_weights = None

        motif_vocabulary = dataset.metadata.get("motif_vocabulary")
        self._uses_motifs = motif_vocabulary is not None

        self._node_categorical_num_classes = len(dataset.node_type_index_to_string)

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in params:
                self._node_categorical_features_embedding = None

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

    def sample_from_latent_repr(self, latent_repr):
        mean_and_log_var = self.mean_log_var_mlp(latent_repr)
        # perturb latent repr
        mu = mean_and_log_var[:, : self.latent_dim]  # Shape: [V, MD]
        log_var = mean_and_log_var[:, self.latent_dim :]  # Shape: [V, MD]

        # result_representations: shape [num_partial_graphs, latent_repr_dim]
        p, q, z = self.sample(mu, log_var)

        return p, q, z

    def sample(self, mu, log_var):
        """Samples a different noise vector for each partial graph.
        TODO: look into the other sampling strategies."""
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward(self, batch):
        # Obtain graph level representation of original molecular graph
        input_molecule_representations = self.full_graph_encoder(
            original_graph_node_categorical_features=batch.original_graph_node_categorical_features,
            node_features=batch.original_graph_x.float(),
            edge_index=batch.original_graph_edge_index,
            edge_type=batch.original_graph_edge_type,
            batch_index=batch.original_graph_x_batch,
        )

        # Obtain graph level representation of the partial graph
        partial_graph_representions, node_representations = self.partial_graph_encoder(
            node_features=batch.x,
            edge_index=batch.edge_index.long(),
            edge_type=batch.edge_type,
            batch_index=batch.batch,
        )

        # Apply latent sampling strategy
        p, q, latent_representation = self.sample_from_latent_repr(
            input_molecule_representations
        )

        # Forward pass through decoder
        (
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
            graph_to_focus_node_map=batch.focus_node,
            node_to_graph_map=batch.batch,
            candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
            candidate_edge_features=batch.edge_features,
            # attachment selection
            candidate_attachment_points=batch.valid_attachment_point_choices.long(),
        )

        # NOTE: loss computation will be done in lightning module
        return MoLeROutput(
            node_type_logits=node_type_logits,
            edge_candidate_logits=edge_candidate_logits,
            edge_type_logits=edge_type_logits,
            attachment_point_selection_logits=attachment_point_selection_logits,
            p=p,
            q=q,
        )

    def compute_loss(self, moler_output, batch):
        num_correct_node_type_choices = (
            batch.correct_node_type_choices_ptr.unique().shape[-1] - 1
        )
        node_type_multihot_labels = batch.correct_node_type_choices.view(
            num_correct_node_type_choices, -1
        )

        loss = self.decoder.compute_decoder_loss(
            node_type_logits=moler_output.node_type_logits,
            node_type_multihot_labels=node_type_multihot_labels,
            num_graphs_in_batch=len(batch.ptr) - 1,
            node_to_graph_map=batch.batch,
            candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
            edge_candidate_logits=moler_output.edge_candidate_logits,  # as is
            per_graph_num_correct_edge_choices=batch.num_correct_edge_choices,
            edge_candidate_correctness_labels=batch.correct_edge_choices,
            no_edge_selected_labels=batch.stop_node_label,
            attachment_point_selection_logits=moler_output.attachment_point_selection_logits,
            attachment_point_candidate_to_graph_map=batch.valid_attachment_point_choices_batch.long(),
            attachment_point_correct_choices=batch.correct_attachment_point_choice.long(),
        )
