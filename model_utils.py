import torch
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.nn import RGATConv
from torch_geometric.nn import aggr
from dataclasses import dataclass


@dataclass
class MoLeROutput:
    first_node_type_logits: torch.Tensor
    node_type_logits: torch.Tensor
    edge_candidate_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    attachment_point_selection_logits: torch.Tensor
    p: torch.Tensor
    q: torch.Tensor


class GenericGraphEncoder(torch.nn.Module):
    """
    Generic Graph Encoder that uses intermediate layer results and outputs graph level
    representation as well as node level representation.
    TODO: add layer norm
    """

    def __init__(
        self,
        input_feature_dim,
        num_relations=4,
        hidden_layer_feature_dim=64,
        num_layers=12,
        layer_type="RGATConv",
        use_intermediate_gnn_results=True,
    ):
        super(GenericGraphEncoder, self).__init__()
        if layer_type == "RGATConv":
            self._first_layer = RGATConv(
                in_channels=input_feature_dim,
                out_channels=hidden_layer_feature_dim,
                num_relations=num_relations,
            )

            self._encoder_layers = torch.nn.ModuleList(
                [
                    RGATConv(
                        in_channels=hidden_layer_feature_dim,
                        out_channels=hidden_layer_feature_dim,
                        num_relations=num_relations,
                    )
                    for _ in range(num_layers)
                ]
            )
            self._softmax_aggr = aggr.SoftmaxAggregation(learn=True)
            self._use_intermediate_gnn_results = use_intermediate_gnn_results
        else:
            raise NotImplementedError

    def forward(self, node_features, edge_index, edge_type, batch_index):
        gnn_results = []
        gnn_results += [self._first_layer(node_features, edge_index.long(), edge_type)]

        for i, layer in enumerate(self._encoder_layers):
            gnn_results += [layer(gnn_results[-1], edge_index.long(), edge_type)]

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


def get_params():
    return {'full_graph_encoder': {'input_feature_dim': 32,
  'atom_or_motif_vocab_size': 139},
 'partial_graph_encoder': {'input_feature_dim': 32},
 'mean_log_var_mlp': {'input_feature_dim': 832, 'output_size': 1024},
 'decoder': {'node_type_selector': {'input_feature_dim': 1344,
   'output_size': 140},
  'node_type_loss_weights': torch.tensor([10.0000,  0.1000,  0.1000,  0.1000,  0.7879,  0.4924,  0.6060, 10.0000,
           7.8786, 10.0000,  7.8786,  0.1000,  0.6565,  0.6565,  0.9848,  0.8754,
           0.8754,  1.1255,  0.9848,  1.3131,  1.5757,  1.9696,  1.5757,  1.9696,
           2.6262,  1.9696,  1.9696,  7.8786,  7.8786,  3.9393,  2.6262,  2.6262,
           2.6262,  2.6262,  3.9393,  7.8786,  7.8786,  7.8786,  3.9393,  7.8786,
          10.0000,  7.8786,  3.9393,  3.9393,  3.9393,  3.9393,  3.9393,  3.9393,
           3.9393,  3.9393,  3.9393,  3.9393,  3.9393,  3.9393,  7.8786,  7.8786,
          10.0000, 10.0000,  7.8786,  7.8786, 10.0000,  7.8786,  7.8786, 10.0000,
           7.8786,  7.8786, 10.0000,  7.8786, 10.0000,  7.8786,  7.8786, 10.0000,
           7.8786,  7.8786,  7.8786, 10.0000, 10.0000,  7.8786,  7.8786,  7.8786,
           7.8786,  7.8786, 10.0000, 10.0000, 10.0000, 10.0000,  7.8786, 10.0000,
          10.0000, 10.0000,  7.8786, 10.0000,  7.8786, 10.0000,  7.8786, 10.0000,
          10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  7.8786,
          10.0000,  7.8786,  7.8786,  7.8786,  7.8786, 10.0000,  7.8786, 10.0000,
          10.0000, 10.0000,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,
           7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,
           7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,  7.8786,
           7.8786,  7.8786,  7.8786,  0.1000]),
  'no_more_edges_repr': (1, 835),
  'edge_candidate_scorer': {'input_feature_dim': 3011, 'output_size': 1},
  'edge_type_selector': {'input_feature_dim': 3011, 'output_size': 3},
  'attachment_point_selector': {'input_feature_dim': 2176, 'output_size': 1},
  'first_node_type_selector': {'input_feature_dim': 512, 'output_size': 139}},
 'latent_sample_strategy': 'per_graph',
 'latent_repr_dim': 512,
 'latent_repr_size': 512}