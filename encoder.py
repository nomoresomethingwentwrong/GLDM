from model_utils import GenericGraphEncoder
import torch
from model_utils import LayerType

class GraphEncoder(torch.nn.Module):
    """Returns graph level representation of the molecules."""

    def __init__(
        self,
        input_feature_dim,
        atom_or_motif_vocab_size,
        motif_embedding_size=64,
        hidden_layer_feature_dim=64,
        num_layers=12,
        layer_type=LayerType.GCNConv,#"RGATConv",
        use_intermediate_gnn_results=True,
    ):
        super(GraphEncoder, self).__init__()
        self._gnn_layer_type = layer_type
        self._dummy_param = torch.nn.Parameter(torch.empty(0)) # for inferrring device of model
        self._embed = torch.nn.Embedding(atom_or_motif_vocab_size, motif_embedding_size)
        self._model = GenericGraphEncoder(
            input_feature_dim=motif_embedding_size + input_feature_dim,
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_layers=num_layers,
            layer_type=self._gnn_layer_type,
            use_intermediate_gnn_results=use_intermediate_gnn_results,
        )

    @property
    def gnn_layer_type(self):
        return self._gnn_layer_type

    def forward(
        self,
        original_graph_node_categorical_features,
        node_features,
        edge_index,
        edge_features, # can be edge type or edge attr
        batch_index,
    ):
        motif_embeddings = self._embed(original_graph_node_categorical_features)
        node_features = torch.cat((node_features, motif_embeddings), axis=-1)
        if self.gnn_layer_type == LayerType.RGATConv:
            edge_type = edge_features.int()
            input_molecule_representations, _ = self._model(
                node_features, edge_index.long(), edge_type, batch_index
            )
        elif self.gnn_layer_type == LayerType.GCNConv:
            edge_attr = edge_features.float()
            input_molecule_representations, _ = self._model(
                node_features, edge_index.long(), edge_attr, batch_index
            )
        else:
            raise not NotImplementedError
        
        return input_molecule_representations


class PartialGraphEncoder(torch.nn.Module):
    """Returns graph level representation of the molecules."""

    def __init__(
        self,
        input_feature_dim,
        atom_or_motif_vocab_size,
        motif_embedding_size=64,
        hidden_layer_feature_dim=64,
        num_layers=12,
        layer_type=LayerType.GCNConv,#"RGATConv",
        use_intermediate_gnn_results=True,
    ):
        super(PartialGraphEncoder, self).__init__()
        self._gnn_layer_type = layer_type
        self._dummy_param = torch.nn.Parameter(torch.empty(0)) # for inferrring device of model
        self._embed = torch.nn.Embedding(atom_or_motif_vocab_size, motif_embedding_size)
        self._model = GenericGraphEncoder(
            input_feature_dim=motif_embedding_size + input_feature_dim + 1, # add one for node in focus bit
            hidden_layer_feature_dim=hidden_layer_feature_dim,
            num_layers=num_layers,
            layer_type=self._gnn_layer_type,
            use_intermediate_gnn_results=use_intermediate_gnn_results,
        )

    @property
    def gnn_layer_type(self):
        return self._gnn_layer_type

    def forward(
        self,
        partial_graph_node_categorical_features,
        node_features,
        edge_index,
        edge_features, # can be edge type or edge attr
        graph_to_focus_node_map,
        candidate_attachment_points,
        batch_index,
    ):
        motif_embeddings = self._embed(partial_graph_node_categorical_features)
        initial_node_features = torch.cat(
            [node_features, motif_embeddings], axis=-1
        )
        node_features = torch.cat((node_features, motif_embeddings), axis=-1)

        nodes_to_set_in_focus_bit = torch.cat(
            [graph_to_focus_node_map, candidate_attachment_points], axis=0
        )

        node_is_in_focus_bit_zeros = torch.zeros((node_features.shape[0], 1), device = self._dummy_param.device)

        node_is_in_focus_bit = node_is_in_focus_bit_zeros.index_add_(
            dim = 0, 
            index = nodes_to_set_in_focus_bit.int().to(self._dummy_param.device), 
            source = torch.ones((nodes_to_set_in_focus_bit.shape[0], 1), device = self._dummy_param.device)
        )
        node_is_in_focus_bit = node_is_in_focus_bit.minimum(torch.ones(1, device = self._dummy_param.device))
        initial_node_features = torch.cat([initial_node_features, node_is_in_focus_bit], axis=-1)

        if self.gnn_layer_type == LayerType.RGATConv:
            edge_type = edge_features.int()
            partial_graph_representions, node_representations = self._model(
                initial_node_features, edge_index.long(), edge_type, batch_index
            )
        elif self.gnn_layer_type == LayerType.GCNConv:
            edge_attr = edge_features.float()
            partial_graph_representions, node_representations = self._model(
                initial_node_features, edge_index.long(), edge_attr, batch_index
            )
        else:
            raise not NotImplementedError

        return partial_graph_representions, node_representations