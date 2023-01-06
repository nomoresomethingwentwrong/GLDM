import torch
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.nn import RGATConv, GCNConv, LayerNorm, Sequential
from torch_geometric.nn import aggr
from dataclasses import dataclass
import sys
from enum import Enum, auto

sys.path.append("../moler_reference")

from molecule_generation.utils.training_utils import get_class_balancing_weights


@dataclass
class MoLeROutput:
    first_node_type_logits: torch.Tensor
    node_type_logits: torch.Tensor
    edge_candidate_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    attachment_point_selection_logits: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


class LayerType(Enum):
    GCNConv = auto()
    RGATConv = auto()


class GenericGraphEncoder(torch.nn.Module):
    """
    Generic Graph Encoder that uses intermediate layer results and outputs graph level
    representation as well as node level representation.
    TODO: add layer norm
    """

    def __init__(
        self,
        input_feature_dim,
        num_relations=3,
        hidden_layer_feature_dim=64,
        num_layers=12,
        layer_type=LayerType.GCNConv,  # "RGATConv",
        use_intermediate_gnn_results=True,
    ):
        super(GenericGraphEncoder, self).__init__()
        self._layer_type = layer_type

        if self._layer_type == LayerType.RGATConv:
            self._first_layer = RGATConv(
                in_channels=input_feature_dim,
                out_channels=hidden_layer_feature_dim,
                num_relations=num_relations,
            )

            self._encoder_layers = torch.nn.ModuleList(
                [
                    Sequential(
                        "x, edge_index, edge_type",
                        [
                            LayerNorm(
                                in_channels=hidden_layer_feature_dim
                            ),  # layer norm before activation as stated here https://www.reddit.com/r/learnmachinelearning/comments/5px958/should_layernorm_be_used_before_or_after_the/
                            LeakyReLU(),
                            (
                                RGATConv(
                                    in_channels=hidden_layer_feature_dim,
                                    out_channels=hidden_layer_feature_dim,
                                    num_relations=num_relations,  # additional parameter for RGATConv
                                ),
                                "x, edge_index, edge_type -> x",
                            ),
                        ],
                    )
                    for _ in range(num_layers)
                ]
            )

            self._softmax_aggr = aggr.SoftmaxAggregation(learn=True)
            self._use_intermediate_gnn_results = use_intermediate_gnn_results

        elif self._layer_type == LayerType.GCNConv:
            self._first_layer = GCNConv(
                in_channels=input_feature_dim,
                out_channels=hidden_layer_feature_dim,
            )

            self._encoder_layers = torch.nn.ModuleList(
                [
                    Sequential(
                        "x, edge_index",
                        [
                            (
                                LayerNorm(in_channels=hidden_layer_feature_dim),
                                "x -> x",
                            ),  # layer norm before activation as stated here https://www.reddit.com/r/learnmachinelearning/comments/5px958/should_layernorm_be_used_before_or_after_the/
                            LeakyReLU(),
                            (
                                GCNConv(
                                    in_channels=hidden_layer_feature_dim,
                                    out_channels=hidden_layer_feature_dim,
                                ),
                                "x, edge_index -> x",
                            ),
                        ],
                    )
                    for _ in range(num_layers)
                ]
            )

            self._softmax_aggr = aggr.SoftmaxAggregation(learn=True)
            self._use_intermediate_gnn_results = use_intermediate_gnn_results

        else:
            raise NotImplementedError

    def forward(self, node_features, edge_index, edge_type_or_attr, batch_index):
        gnn_results = []

        if self._layer_type == LayerType.RGATConv:

            gnn_results += [
                self._first_layer(node_features, edge_index.long(), edge_type_or_attr)
            ]

            for layer in self._encoder_layers:
                gnn_results += [
                    layer(gnn_results[-1], edge_index.long(), edge_type_or_attr)
                ]

        elif "GCNConv" in str(
            self._layer_type
        ):  # self._layer_type == LayerType.GCNConv: # GCNConv does not require edge features or edge attrs

            gnn_results += [self._first_layer(node_features, edge_index.long())]

            for layer in self._encoder_layers:
                gnn_results += [layer(gnn_results[-1], edge_index.long())]

        if self._use_intermediate_gnn_results:
            x = torch.cat(gnn_results, axis=-1)
            graph_representations = self._softmax_aggr(x, batch_index)

        else:
            graph_representations = self._softmax_aggr(gnn_results[-1], batch_index)
        node_representations = torch.cat(gnn_results, axis=-1)
        return graph_representations, node_representations


class GenericMLP(torch.nn.Module):
    """
    Generic MLP with dropout layers.
    """

    def __init__(
        self,
        input_feature_dim,
        output_size,
        hidden_layer_feature_dim=64,
        num_hidden_layers=1,
        activation_layer_type="leaky_relu",
        dropout_prob=0.2,
    ):
        super(GenericMLP, self).__init__()
        if activation_layer_type == "leaky_relu":
            self._first_layer = Linear(input_feature_dim, hidden_layer_feature_dim)
            self._hidden_layers = torch.nn.ModuleList(
                [
                    LeakyReLU(),
                    Dropout(p=dropout_prob),
                    Linear(hidden_layer_feature_dim, hidden_layer_feature_dim),
                ]
                * num_hidden_layers
            )
            self._output_layer = torch.nn.Sequential(
                LeakyReLU(),
                Dropout(p=dropout_prob),
                Linear(hidden_layer_feature_dim, output_size),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self._first_layer(x)
        for layer in self._hidden_layers:
            x = layer(x)
        x = self._output_layer(x)
        return x


def get_class_weights(dataset, class_weight_factor=1.0):
    next_node_type_distribution = dataset.metadata.get(
        "train_next_node_type_distribution"
    )
    atom_type_distribution = dataset.metadata.get("train_atom_type_distribution")
    num_node_types = dataset.num_node_types
    atom_type_nums = [
        atom_type_distribution[dataset.node_type_index_to_string[type_idx]]
        for type_idx in range(num_node_types)
    ]
    atom_type_nums.append(next_node_type_distribution["None"])
    class_weights = get_class_balancing_weights(
        class_counts=atom_type_nums, class_weight_factor=class_weight_factor
    )
    return class_weights


def get_params(dataset):
    return {
        "full_graph_encoder": {
            "input_feature_dim": dataset[0].x.shape[-1],
            "atom_or_motif_vocab_size": len(dataset.node_type_index_to_string),
        },
        "partial_graph_encoder": {
            "input_feature_dim": dataset[0].x.shape[-1],
            "atom_or_motif_vocab_size": len(dataset.node_type_index_to_string),
        },
        "mean_log_var_mlp": {"input_feature_dim": 832, "output_size": 1024},
        "decoder": {
            "node_type_selector": {
                "input_feature_dim": 1344,
                "output_size": len(dataset.node_type_index_to_string) + 1,
            },
            "node_type_loss_weights": torch.tensor(get_class_weights(dataset)).cuda(),
            "no_more_edges_repr": (1, 835),
            "edge_candidate_scorer": {"input_feature_dim": 3011, "output_size": 1},
            "edge_type_selector": {"input_feature_dim": 3011, "output_size": 3},
            "attachment_point_selector": {"input_feature_dim": 2176, "output_size": 1},
            "first_node_type_selector": {
                "input_feature_dim": 512,
                "output_size": len(dataset.node_type_index_to_string),
            },
        },
        "latent_sample_strategy": "per_graph",
        "latent_repr_dim": 512,
        "latent_repr_size": 512,
        "kl_divergence_weight":0.02,
        "kl_divergence_annealing_beta":0.999,
        "training_hyperparams": {
            "max_lr": 1e-2,
            "div_factor": 25,
            "three_phase": True,
        },
    }
