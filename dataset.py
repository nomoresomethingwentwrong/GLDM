from torch_geometric.data import Dataset, Data
import os
import pandas as pd
import numpy as np
import torch
import gzip
import pickle
import concurrent.futures
import random
import sys

sys.path.append("../moler_reference")


class MolerData(Data):
    """To ensure that both the original graph and the partial graph edge indices are incremented."""

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        original_graph_edge_index=None,
        original_graph_x=None,
        valid_attachment_point_choices=None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.x = x
        self.original_graph_edge_index = original_graph_edge_index
        self.original_graph_x = original_graph_x
        self.valid_attachment_point_choices = valid_attachment_point_choices

    def __inc__(self, key, value, *args, **kwargs):
        if key == "original_graph_edge_index":
            return self.original_graph_x.size(0)
        if key == "focus_node":
            return self.x.size(0)
        if key == "correct_attachment_point_choice":
            return self.valid_attachment_point_choices.size(0)
        if key == "valid_attachment_point_choices":
            return self.x.size(0)
        if key == "valid_edge_choices":
            return self.x.size(0)
        if key == 'candidate_attachment_points':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'latent_representation':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def get_motif_type_to_node_type_index_map(motif_vocabulary, num_atom_types):
    """Helper to construct a mapping from motif type to shifted node type."""

    return {
        motif: num_atom_types + motif_type
        for motif, motif_type in motif_vocabulary.vocabulary.items()
    }


class MolerDataset(Dataset):
    def __init__(
        self,
        root,
        raw_moler_trace_dataset_parent_folder,  # absolute path
        output_pyg_trace_dataset_parent_folder,  # absolute path
        split="train",
        transform=None,
        pre_transform=None,
        using_self_loops=True,
        gen_step_drop_probability=0.0,
    ):
        self._processed_file_paths = None
        self._transform = transform
        self._pre_transform = pre_transform
        self._raw_moler_trace_dataset_parent_folder = (
            raw_moler_trace_dataset_parent_folder
        )
        self._output_pyg_trace_dataset_parent_folder = (
            output_pyg_trace_dataset_parent_folder
        )
        self._split = split
        self._using_self_loops = using_self_loops

        # attribute for current molecule ie current data shard
        self._current_in_memory_data_shard = None
        self._current_in_memory_data_shard_counter = None

        self._gen_step_drop_probability = gen_step_drop_probability
        self.load_metadata()

        # create the directory for the processed data if it doesn't exist
        processed_file_paths_folder = os.path.join(
            self._output_pyg_trace_dataset_parent_folder, self._split
        )
        if not os.path.exists(processed_file_paths_folder):
            os.mkdir(processed_file_paths_folder)
        # try to read in the csv with the processed file paths
        processed_file_paths_csv = os.path.join(
            processed_file_paths_folder, "processed_file_paths.csv"
        )
        if os.path.exists(processed_file_paths_csv):
            self._processed_file_paths = pd.read_csv(processed_file_paths_csv)[
                "file_names"
            ].tolist()

        super().__init__(root, transform, pre_transform)
        self._processed_file_paths = pd.read_csv(processed_file_paths_csv)[
            "file_names"
        ].tolist()

    @staticmethod
    def _generate_self_loops(num_nodes):
        """Generate a (num_nodes, 2) array of self loop edges."""
        return np.repeat(np.arange(num_nodes, dtype=np.int32), 2).reshape(-1, 2)

    @property
    def raw_file_names(self):
        """
        Raw generation trace files output from the preprocess function of the cli. These are zipped pickle
        files. This is the actual file name without the parent folder.
        """
        raw_pkl_file_folders = [
            folder
            for folder in os.listdir(self._raw_moler_trace_dataset_parent_folder)
            if folder.startswith(self._split)
        ]

        assert (
            len(raw_pkl_file_folders) > 0
        ), f"{self._raw_moler_trace_dataset_parent_folder} does not contain {self._split} files."

        raw_generation_trace_files = []
        for folder in raw_pkl_file_folders:
            for pkl_file in os.listdir(
                os.path.join(self._raw_moler_trace_dataset_parent_folder, folder)
            ):
                raw_generation_trace_files.append(
                    os.path.join(
                        self._raw_moler_trace_dataset_parent_folder, folder, pkl_file
                    )
                )
        return raw_generation_trace_files

    @property
    def processed_file_names(self):
        """Processed generation trace objects that are stored as .pt files"""
        if self._processed_file_paths is not None:
            return self._processed_file_paths
        else:
            return []

    @property
    def processed_file_names_size(self):
        return len(self.processed_file_names)

    @property
    def metadata(self):
        return self._metadata

    @property
    def node_type_index_to_string(self):
        return self._node_type_index_to_string

    @property
    def num_node_types(self):
        return len(self.node_type_index_to_string)

    def node_type_to_index(self, node_type):
        return self._atom_type_featuriser.type_name_to_index(node_type)

    def node_types_to_indices(self, node_types):
        """Convert list of string representations into list of integer indices."""
        return [self.node_type_to_index(node_type) for node_type in node_types]

    def node_types_to_multi_hot(self, node_types):
        """Convert between string representation to multi hot encoding of correct node types.
        """
        correct_indices = self.node_types_to_indices(node_types)
        multihot = np.zeros(shape=(self.num_node_types,), dtype=np.float32)
        for idx in correct_indices:
            multihot[idx] = 1.0
        return multihot

    def node_type_to_index(self, node_type):
        motif_node_type_index = self._motif_to_node_type_index.get(node_type)

        if motif_node_type_index is not None:
            return motif_node_type_index
        else:
            return self._atom_type_featuriser.type_name_to_index(node_type)

    def load_metadata(self):
        metadata_file_path = os.path.join(
            self._raw_moler_trace_dataset_parent_folder, "metadata.pkl.gz"
        )

        with gzip.open(metadata_file_path, "rb") as f:
            self._metadata = pickle.load(f)

        self._atom_type_featuriser = next(
            featuriser
            for featuriser in self._metadata["feature_extractors"]
            if featuriser.name == "AtomType"
        )

        self._node_type_index_to_string = (
            self._atom_type_featuriser.index_to_atom_type_map.copy()
        )
        self._motif_vocabulary = self.metadata.get("motif_vocabulary")

        if self._motif_vocabulary is not None:
            self._motif_to_node_type_index = get_motif_type_to_node_type_index_map(
                motif_vocabulary=self._motif_vocabulary,
                num_atom_types=len(self._node_type_index_to_string),
            )

            for motif, node_type in self._motif_to_node_type_index.items():
                self._node_type_index_to_string[node_type] = motif
        else:
            self._motif_to_node_type_index = {}

    def generate_preprocessed_file_paths_csv(
        self, preprocessed_file_paths_folder, results
    ):
        # file_paths = [
        #     os.path.join(preprocessed_file_paths_folder, file_path)
        #     for file_path in os.listdir(preprocessed_file_paths_folder)
        # ]
        file_paths = [result["file_path"] for result in results]
        molecule_gen_steps_length = [
            result["molecule_gen_steps_length"] for result in results
        ]
        df = pd.DataFrame(
            {
                "file_names": file_paths,
                "molecule_gen_steps_length": molecule_gen_steps_length,
            }
        )
        processed_file_paths_csv = os.path.join(
            preprocessed_file_paths_folder, "processed_file_paths.csv"
        )
        df.to_csv(processed_file_paths_csv, index=False)

    def process(self):
        """Convert raw generation traces into individual .pt files for each of the trace steps."""
        # only call process if it was not called before
        if self.processed_file_names_size > 0:
            pass
        else:
            self.load_metadata()
            for pkl_file_path in self.raw_file_names:
                generation_steps = self._convert_data_shard_to_list_of_trace_steps(
                    pkl_file_path
                )

                # for molecule_idx, molecule_gen_steps in generation_steps:
                with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                    futures = executor.map(
                        self._save_processed_gen_step,
                        (
                            (pkl_file_path, molecule_idx, molecule_gen_steps)
                            for molecule_idx, molecule_gen_steps in generation_steps
                        ),
                    )
                    results = list(futures)
                    # for future in futures:
                    #     print(f"Done with molecule {future}")
                    # for step_idx, step in enumerate(molecule_gen_steps):

            self.generate_preprocessed_file_paths_csv(
                preprocessed_file_paths_folder=os.path.join(
                    self._output_pyg_trace_dataset_parent_folder, self._split
                ),
                results=results,
            )

    def _save_processed_gen_step(self, pkl_file_path_molecule_idx_molecule_gen_steps):
        (
            pkl_file_path,
            molecule_idx,
            molecule_gen_steps,
        ) = pkl_file_path_molecule_idx_molecule_gen_steps

        # for step_idx, step in enumerate(molecule_gen_steps):
        #     file_name = f'{pkl_file_path.split("/")[-1].split(".")[0]}_mol_{molecule_idx}_step_{step_idx}.pt'  # .pkl.gz'  #
        #     file_path = os.path.join(
        #         self._output_pyg_trace_dataset_parent_folder,
        #         self._split,
        #         file_name,
        #     )
        #     print("file_path", file_path)
        #     torch.save(step, file_path)
        #     print(f"Processing {molecule_idx}, step {step_idx}")

        file_name = (
            f'{pkl_file_path.split("/")[-1].split(".")[0]}_mol_{molecule_idx}.pkl.gz'  #
        )
        file_path = os.path.join(
            self._output_pyg_trace_dataset_parent_folder,
            self._split,
            file_name,
        )
        with gzip.open(file_path, "wb") as shard_file_path:
            pickle.dump(molecule_gen_steps, shard_file_path)

        return {
            "file_path": file_path,
            "molecule_gen_steps_length": len(molecule_gen_steps),
        }

    def _convert_data_shard_to_list_of_trace_steps(self, pkl_file_path):
        # TODO: multiprocessing to speed this up
        generation_steps = []

        with gzip.open(pkl_file_path, "rb") as f:
            molecules = pickle.load(f)
            for molecule_idx, molecule in enumerate(molecules):

                generation_steps += [
                    (molecule_idx, self._extract_generation_steps(molecule))
                ]

        return generation_steps

    def _extract_generation_steps(self, molecule):
        molecule_gen_steps = []
        molecule_property_values = {
            k: [v] for k, v in molecule.graph_property_values.items()
        }
        for gen_step in molecule:
            gen_step_features = {}

            gen_step_features["original_graph_x"] = molecule.node_features
            # have an edge type attribute to tell apart each of the 3 bond types
            edge_indexes = []
            edge_types = []
            for i, adj_list in enumerate(molecule.adjacency_lists):
                if len(adj_list) != 0:
                    edge_index = adj_list.T
                    edge_indexes += [edge_index]
                    """ 
                    edge types: 
                    single bond => 0
                    double bond => 1
                    triple bond => 2
                    self loop => 3
                    """
                    edge_types += [i] * len(adj_list)

            # add self loops
            if self._using_self_loops:
                num_nodes_in_original_graph = molecule.node_features.shape[0]
                edge_indexes += [
                    self._generate_self_loops(num_nodes=num_nodes_in_original_graph).T
                ]
                edge_types += [3] * num_nodes_in_original_graph

            gen_step_features["original_graph_edge_index"] = (
                np.concatenate(edge_indexes, 1)
                if len(edge_indexes) > 0
                else np.array(edge_indexes)
            )
            gen_step_features["original_graph_edge_type"] = np.array(edge_types)
            gen_step_features[
                "original_graph_node_categorical_features"
            ] = molecule.node_categorical_features

            gen_step_features["x"] = gen_step.partial_node_features
            gen_step_features["focus_node"] = [gen_step.focus_node]

            # have an edge type attribute to tell apart each of the 3 bond types
            edge_indexes = []
            edge_types = []
            for i, adj_list in enumerate(gen_step.partial_adjacency_lists):
                if len(adj_list) != 0:
                    edge_index = adj_list.T
                    edge_indexes += [edge_index]
                    """ 
                    edge types: 
                    single bond => 0
                    double bond => 1
                    triple bond => 2
                    self loop => 3
                    """
                    edge_types += [i] * len(adj_list)

            # add self loops
            if self._using_self_loops:
                num_nodes_in_partial_graph = gen_step.partial_node_features.shape[0]
                edge_indexes += [
                    self._generate_self_loops(num_nodes=num_nodes_in_partial_graph).T
                ]
                edge_types += [3] * num_nodes_in_partial_graph

            gen_step_features["edge_index"] = (
                np.concatenate(edge_indexes, 1)
                if len(edge_indexes) > 0
                else np.array(edge_indexes)
            )
            gen_step_features["edge_type"] = np.array(edge_types)
            gen_step_features["edge_features"] = np.array(gen_step.edge_features)
            gen_step_features["correct_edge_choices"] = gen_step.correct_edge_choices

            num_correct_edge_choices = np.sum(gen_step.correct_edge_choices)
            gen_step_features["num_correct_edge_choices"] = [num_correct_edge_choices]
            gen_step_features["stop_node_label"] = [int(num_correct_edge_choices == 0)]
            gen_step_features["valid_edge_choices"] = gen_step.valid_edge_choices
            gen_step_features["valid_edge_types"] = gen_step.valid_edge_types

            gen_step_features["correct_edge_types"] = gen_step.correct_edge_types
            gen_step_features[
                "partial_node_categorical_features"
            ] = gen_step.partial_node_categorical_features
            if gen_step.correct_attachment_point_choice is not None:
                gen_step_features["correct_attachment_point_choice"] = [
                    list(gen_step.valid_attachment_point_choices).index(
                        gen_step.correct_attachment_point_choice
                    )
                ]
            else:
                gen_step_features["correct_attachment_point_choice"] = []
            gen_step_features[
                "valid_attachment_point_choices"
            ] = gen_step.valid_attachment_point_choices

            # And finally, the correct node type choices. Here, we have an empty list of
            # correct choices for all steps where we didn't choose a node, so we skip that:
            if gen_step.correct_node_type_choices is not None:
                gen_step_features[
                    "correct_node_type_choices"
                ] = np.array([self.node_types_to_multi_hot(gen_step.correct_node_type_choices)])
            else:
                gen_step_features["correct_node_type_choices"] = np.zeros(shape = (0,) + (self.num_node_types, ))
            if molecule.correct_first_node_type_choices is not None:
                gen_step_features[
                    "correct_first_node_type_choices"
                ] = np.array([self.node_types_to_multi_hot(molecule.correct_first_node_type_choices)])
            else:
                gen_step_features["correct_first_node_type_choices"] = np.zeros(shape = (0,) + (self.num_node_types, ))
            # Add graph_property_values
            gen_step_features = {**gen_step_features, **molecule_property_values}
            molecule_gen_steps += [gen_step_features]

        molecule_gen_steps = self._to_tensor_moler(molecule_gen_steps)

        return [MolerData(**step) for step in molecule_gen_steps]

    def _to_tensor_moler(self, molecule_gen_steps):
        for i in range(len(molecule_gen_steps)):
            for k, v in molecule_gen_steps[i].items():
                molecule_gen_steps[i][k] = torch.tensor(molecule_gen_steps[i][k])
        return molecule_gen_steps

    def len(self):
        return self.processed_file_names_size

    def _size_of_current_data_shard(self):
        return len(self._current_in_memory_data_shard)

    def _reached_end_of_current_data_shard(self):
        return self._current_in_memory_data_shard_counter == len(
            self._current_in_memory_data_shard
        )

    def _read_in_data_shard(self, file_path):
        with gzip.open(file_path, "rb") as f:
            self._current_in_memory_data_shard = pickle.load(f)
        self._current_in_memory_data_shard_counter = 0

    def _read_in_and_extract_first_item_of_data_shard(self, file_path):
        self._read_in_data_shard(file_path)
        self._current_in_memory_data_shard_counter += 1  # reset counter
        return self._current_in_memory_data_shard[0]

    def _drop_current_gen_step(self):
        return True if random.random() < self._gen_step_drop_probability else False

    def get(self, idx):
        """
        This is a workaround for reading in one data shard at a time (one molecule at a time)
        Each molecule has a varying number of generation steps, and all the generation steps
        for one particular molecule will be stored in a single data shard.

        We use the idx to reference the molecule idx and read in each data shard and store it
        as an attribute in the class. Then, we maintain a counter for iterating through the
        shard. Once we read the end of the data shard, we use the idx to read in another molecule.
        """
        if self._current_in_memory_data_shard is not None:
            while not self._reached_end_of_current_data_shard():
                # decide whether to drop the current generation step just like in MoLeR

                if not self._drop_current_gen_step():
                    data = self._current_in_memory_data_shard[
                        self._current_in_memory_data_shard_counter
                    ]
                    self._current_in_memory_data_shard_counter += 1
                    return data

                self._current_in_memory_data_shard_counter += 1

            file_path = self.processed_file_names[idx]
            return self._read_in_and_extract_first_item_of_data_shard(file_path)
        else:

            file_path = self.processed_file_names[idx]
            return self._read_in_and_extract_first_item_of_data_shard(file_path)

        # alternative for reading in individual .pt files (NOTE currently infeasible)
        # file_path = self.processed_file_names[idx]
        # data = torch.load(file_path)
        # return data
