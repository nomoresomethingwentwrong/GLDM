import torch
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.nn import RGATConv
from torch_geometric.nn import aggr


class GenericGraphEncoder(torch.nn.Module):
    """
    Generic Graph Encoder that uses intermediate layer results and outputs graph level
    representation as well as node level representation.
    TODO: add layer norm
    """

    def __init__(
        self,
        input_feature_dim,
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
                num_relations=3,
            )

            self._encoder_layers = torch.nn.ModuleList(
                [
                    RGATConv(
                        in_channels=hidden_layer_feature_dim,
                        out_channels=hidden_layer_feature_dim,
                        num_relations=3,
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
    TODO: add
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