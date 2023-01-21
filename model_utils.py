import torch
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.nn import (
    FiLMConv,
    RGATConv,
    GATConv,
    GCNConv,
    RGCNConv,
    LayerNorm,
    Sequential,
)
from torch_geometric.nn import aggr
from dataclasses import dataclass
import sys
from enum import Enum, auto
from utils import unsorted_segment_softmax
from torch_geometric.utils import scatter

sys.path.append("../moler_reference")

from molecule_generation.utils.training_utils import get_class_balancing_weights


@dataclass
class MoLeROutput:
    first_node_type_logits: torch.Tensor
    node_type_logits: torch.Tensor
    edge_candidate_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    attachment_point_selection_logits: torch.Tensor
    p: torch.Tensor
    q: torch.Tensor


class LayerType(Enum):
    FiLMConv = auto()  # original MoLeR paper GNN layer
    GCNConv = auto()
    RGCNConv = auto()
    GATConv = auto()
    RGATConv = auto()


class AggrLayerType(Enum):
    MoLeRAggregation = auto()
    SoftmaxAggregation = auto()


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
        layer_type=LayerType.FiLMConv,  # "RGATConv",
        use_intermediate_gnn_results=True,
        aggr_layer_type='MoLeRAggregation',
        total_num_moler_aggr_heads=None,  # half will have sigmoid scoring function, half will have softmax scoring functions
    ):
        super(GenericGraphEncoder, self).__init__()
        self._layer_type = layer_type

        self._first_layer, self._encoder_layers = get_encoder_layers(
            layer_type=layer_type,
            num_layers=num_layers,
            input_feature_dim=input_feature_dim,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_relations=num_relations,
        )
        self._aggr_layer_type = AggrLayerType[aggr_layer_type]
        if self._aggr_layer_type == AggrLayerType.SoftmaxAggregation:
            self._aggr = aggr.SoftmaxAggregation(learn=True)
        elif self._aggr_layer_type == AggrLayerType.MoLeRAggregation:
            assert total_num_moler_aggr_heads is not None
            self._softmax_aggr = WeightedSumGraphRepresentation(
                input_feature_dim=hidden_layer_feature_dim * (num_layers + 1),
                num_heads=total_num_moler_aggr_heads // 2,
                graph_representation_size=hidden_layer_feature_dim
                * (num_layers + 1)
                // 2,
                weighting_fun="softmax",
            )
            self._sigmoid_aggr = WeightedSumGraphRepresentation(
                input_feature_dim=hidden_layer_feature_dim * (num_layers + 1),
                num_heads=total_num_moler_aggr_heads // 2,
                graph_representation_size=hidden_layer_feature_dim
                * (num_layers + 1)
                // 2,
                weighting_fun="sigmoid",
            )
        self._use_intermediate_gnn_results = use_intermediate_gnn_results

    def _apply_aggr(self, x, batch_index):
        if self._aggr_layer_type == AggrLayerType.SoftmaxAggregation:
            return self._aggr(x, batch_index)
        elif self._aggr_layer_type == AggrLayerType.MoLeRAggregation:
            assert hasattr(self, "_softmax_aggr") and hasattr(self, "_sigmoid_aggr")
            weighted_avg = self._softmax_aggr(x, batch_index)
            weighted_sum = self._sigmoid_aggr(x, batch_index)
            return torch.cat(
                (weighted_avg, weighted_sum), axis=-1
            )  # shape should be [batch_size, 832]
        else:
            raise NotImplementedError

    def forward(self, node_features, edge_index, edge_type_or_attr, batch_index):
        gnn_results = []
        if any(
            layer_type in str(self._layer_type)
            for layer_type in ["FiLMConv", "RGATConv", "RGCNConv", "GATConv"]
        ):

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
        else:
            raise NotImplementedError

        if self._use_intermediate_gnn_results:
            x = torch.cat(gnn_results, axis=-1)
            graph_representations = self._apply_aggr(x, batch_index)

        else:
            graph_representations = self._apply_aggr(gnn_results[-1], batch_index)

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
        hidden_layer_feature_dim=256,
        num_hidden_layers=1,
        activation_layer_type="leaky_relu",
        dropout_prob=0.1,
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


class WeightedSumGraphRepresentation(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        input_feature_dim=832,
        graph_representation_size=416,
        weighting_fun="softmax",  # One of {"softmax", "sigmoid"}
        # scoring_mlp_layers = [128],
        scoring_mlp_activation_fun="leaky_relu",
        # scoring_mlp_use_biases: bool = False,
        scoring_mlp_dropout_rate=0.1,
        # transformation_mlp_layers = [128],
        transformation_mlp_activation_fun="leaky_relu",
        # transformation_mlp_use_biases = False,
        transformation_mlp_dropout_rate=0.1,
        #         transformation_mlp_result_lower_bound = None,
        #         transformation_mlp_result_upper_bound = None,
        #         **kwargs,
    ):
        super(WeightedSumGraphRepresentation, self).__init__()
        self._num_heads = num_heads
        self._graph_representation_size = graph_representation_size
        self._weighting_fun = weighting_fun.lower()
        assert self._weighting_fun in ["softmax", "sigmoid"]
        self._transformation_mlp_activation_fun = LeakyReLU()
        self._scoring_mlp = GenericMLP(
            input_feature_dim=input_feature_dim,
            output_size=self._num_heads,  # one score for each head
            hidden_layer_feature_dim=256,
            num_hidden_layers=1,
            activation_layer_type=scoring_mlp_activation_fun,
            dropout_prob=scoring_mlp_dropout_rate,
        )
        self._transformation_mlp = GenericMLP(
            input_feature_dim=input_feature_dim,
            output_size=self._graph_representation_size,  # one score for each head
            hidden_layer_feature_dim=256,
            num_hidden_layers=1,
            activation_layer_type=transformation_mlp_activation_fun,
            dropout_prob=transformation_mlp_dropout_rate,
        )

    def forward(
        self,
        x,
        batch,  # node to graph mapping in a batch
    ):
        # (1) compute weights for each node/head pair:
        scores = self._scoring_mlp(x)  # Shape [number of nodes, number of heads]
        if self._weighting_fun == "sigmoid":
            weights = torch.sigmoid(scores)
        elif self._weighting_fun == "softmax":
            weights_per_head = []
            for head_idx in range(self._num_heads):
                head_scores = scores[:, head_idx]  # Shape [V]
                head_weights = unsorted_segment_softmax(
                    logits=head_scores, segment_ids=batch
                )  # Shape [V]
                weights_per_head.append(head_weights)
            weights = torch.stack(weights_per_head, dim=-1)  # Shape [V, H]
        else:
            raise NotImplementedError()

        # (2) compute representations for each node/head pair:
        node_reprs = self._transformation_mlp_activation_fun(
            self._transformation_mlp(x)
        )
        # Shape [V, graph representation dimension]

        node_reprs = node_reprs.view(
            -1, self._num_heads, self._graph_representation_size // self._num_heads
        )

        # (3) if necessary, weight representations and aggregate by graph:
        weights = torch.unsqueeze(weights, -1)  # Shape [V, H, 1]
        weighted_node_reprs = weights * node_reprs  # Shape [V, H, GD//H]

        weighted_node_reprs = weighted_node_reprs.view(
            -1, self._graph_representation_size
        )
        # Shape [V, GD]
        graph_reprs = scatter(weighted_node_reprs, batch, reduce="sum")  # Shape [G, GD]
        return graph_reprs


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


def get_encoder_layers(
    layer_type,
    num_layers,
    input_feature_dim,
    hidden_layer_feature_dim,
    num_relations=None,
    act=LeakyReLU(),
):
    """Get first layer and encoder layers"""
    if layer_type == LayerType.FiLMConv:
        first_layer = FiLMConv(
            in_channels=input_feature_dim,
            out_channels=hidden_layer_feature_dim,
            num_relations=num_relations,
            act=act,
        )
        encoder_layers = _get_encoder_layers(
            layer_type=LayerType.FiLMConv,
            num_layers=num_layers,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_relations=num_relations,
            act=act,
        )
    elif layer_type == LayerType.RGATConv:
        first_layer = RGATConv(
            in_channels=input_feature_dim,
            out_channels=hidden_layer_feature_dim,
            num_relations=num_relations,
        )
        encoder_layers = _get_encoder_layers(
            layer_type=LayerType.RGATConv,
            num_layers=num_layers,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_relations=num_relations,
        )

    elif layer_type == LayerType.RGCNConv:
        first_layer = RGCNConv(
            in_channels=input_feature_dim,
            out_channels=hidden_layer_feature_dim,
            num_relations=num_relations,
        )
        encoder_layers = _get_encoder_layers(
            layer_type=LayerType.RGCNConv,
            num_layers=num_layers,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_relations=num_relations,
        )

    elif layer_type == LayerType.GATConv:
        first_layer = GATConv(
            in_channels=input_feature_dim,
            out_channels=hidden_layer_feature_dim,
        )
        encoder_layers = _get_encoder_layers(
            layer_type=LayerType.GATConv,
            num_layers=num_layers,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
        )

    elif layer_type == LayerType.GCNConv:
        first_layer = GCNConv(
            in_channels=input_feature_dim,
            out_channels=hidden_layer_feature_dim,
        )
        encoder_layers = _get_encoder_layers(
            layer_type=LayerType.GCNConv,
            num_layers=num_layers,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
        )
    else:
        raise NotImplementedError

    return first_layer, encoder_layers


def _get_encoder_layers(
    layer_type,
    num_layers,
    hidden_layer_feature_dim,
    num_relations=None,
    act=LeakyReLU(),
):
    """Instantiates all layers other than the first"""
    if layer_type == LayerType.FiLMConv:
        assert num_relations is not None
        return torch.nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_type",
                    [
                        (
                            LayerNorm(in_channels=hidden_layer_feature_dim),
                            "x -> x",
                        ),  # layer norm before activation as stated here https://www.reddit.com/r/learnmachinelearning/comments/5px958/should_layernorm_be_used_before_or_after_the/
                        # LeakyReLU(), # remove this because FiLMConv has activation already applied and the example in https://github.com/pyg-team/pytorch_geometric/blob/master/examples/film.py doesnt use it
                        (
                            FiLMConv(
                                in_channels=hidden_layer_feature_dim,
                                out_channels=hidden_layer_feature_dim,
                                num_relations=num_relations,  # additional parameter for RGATConv
                                act=act,
                            ),
                            "x, edge_index, edge_type -> x",
                        ),
                    ],
                )
                for _ in range(num_layers)
            ]
        )
    elif layer_type == LayerType.GCNConv:
        return torch.nn.ModuleList(
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
    elif layer_type == LayerType.RGATConv:
        assert num_relations is not None
        return torch.nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_type",
                    [
                        (
                            LayerNorm(in_channels=hidden_layer_feature_dim),
                            "x -> x",
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
    elif layer_type == LayerType.RGCNConv:
        assert num_relations is not None
        return torch.nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_type",
                    [
                        (
                            LayerNorm(in_channels=hidden_layer_feature_dim),
                            "x -> x",
                        ),  # layer norm before activation as stated here https://www.reddit.com/r/learnmachinelearning/comments/5px958/should_layernorm_be_used_before_or_after_the/
                        LeakyReLU(),
                        (
                            RGCNConv(
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
    elif layer_type == LayerType.GATConv:
        return torch.nn.ModuleList(
            [
                Sequential(
                    "x, edge_index, edge_attr",
                    [
                        (
                            LayerNorm(in_channels=hidden_layer_feature_dim),
                            "x -> x",
                        ),  # layer norm before activation as stated here https://www.reddit.com/r/learnmachinelearning/comments/5px958/should_layernorm_be_used_before_or_after_the/
                        LeakyReLU(),
                        (
                            GATConv(
                                in_channels=hidden_layer_feature_dim,
                                out_channels=hidden_layer_feature_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                    ],
                )
                for _ in range(num_layers)
            ]
        )


def get_params(dataset):
    return {
        "full_graph_encoder": {
            "input_feature_dim": dataset[0].x.shape[-1],
            "atom_or_motif_vocab_size": len(dataset.node_type_index_to_string),
            "aggr_layer_type": "MoLeRAggregation",
            "total_num_moler_aggr_heads": 32, 
        },
        "partial_graph_encoder": {
            "input_feature_dim": dataset[0].x.shape[-1],
            "atom_or_motif_vocab_size": len(dataset.node_type_index_to_string),
            "aggr_layer_type": "MoLeRAggregation",
            "total_num_moler_aggr_heads": 16, 
        },
        "mean_log_var_mlp": {"input_feature_dim": 832, "output_size": 1024},
        "decoder": {
            "node_type_selector": {
                "input_feature_dim": 1344,
                "output_size": len(dataset.node_type_index_to_string) + 1,
            },
            'use_node_type_loss_weights': False, # DON'T use node type loss weights by default
            "node_type_loss_weights": torch.tensor(get_class_weights(dataset)),
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
        "kl_divergence_weight": 0.5,
        "kl_divergence_annealing_beta": 0.999,
        "training_hyperparams": {
            "max_lr": 5e-3,
            "div_factor": 10,
            "three_phase": True,
        },
    }
