from model_utils import GenericGraphEncoder
import torch


class GraphEncoder(torch.nn.Module):
    """Returns graph level representation of the molecules."""

    def __init__(
        self,
        input_feature_dim,
        atom_or_motif_vocab_size,
        motif_embedding_size=64,
        hidden_layer_feature_dim=64,
        num_layers=12,
        layer_type="RGATConv",
        use_intermediate_gnn_results=True,
    ):
        super(GraphEncoder, self).__init__()
        self._embed = torch.nn.Embedding(atom_or_motif_vocab_size, motif_embedding_size)
        self._model = GenericGraphEncoder(
            input_feature_dim=motif_embedding_size + input_feature_dim,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            use_intermediate_gnn_results=use_intermediate_gnn_results,
        )

    def forward(
        self,
        original_graph_node_categorical_features,
        node_features,
        edge_index,
        edge_type,
        batch_index,
    ):
        motif_embeddings = self._embed(original_graph_node_categorical_features)
        node_features = torch.cat((node_features, motif_embeddings), axis=-1)
        input_molecule_representations, _ = self._model(
            node_features, edge_index.long(), edge_type, batch_index
        )
        return input_molecule_representations
