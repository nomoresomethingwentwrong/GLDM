from torch_geometric.data import Dataset, Data, Batch
import os
import pandas as pd
import numpy as np
import torch
import gzip
import pickle
import concurrent.futures
import random
import sys
from tqdm import tqdm
from enum import Enum, auto
from numpy import load


class EdgeRepresentation(Enum):
    edge_attr = auto()
    edge_type = auto()


sys.path.append("../moler_reference")

to_increment_by_num_nodes_in_graph = [
    "focus_node",
    "valid_attachment_point_choices",
    "valid_edge_choices",
    # pick attachment points
    "candidate_attachment_points",
    # pick edge
    "focus_atoms",
    "prior_focus_atoms",
    "candidate_edge_targets",
    # "candidate_edge_type_masks",
]


def chunk_list(elems, chunk_size):
    for i in range(0, len(elems), chunk_size):
        yield elems[i : i + chunk_size]


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
        if key == "correct_attachment_point_choice":
            return self.valid_attachment_point_choices.size(0)
        if key in to_increment_by_num_nodes_in_graph:
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "latent_representation":
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
        using_self_loops=False,
        gen_step_drop_probability=0.5,
        edge_repr=EdgeRepresentation.edge_attr,
        num_samples_debug_mode=None,  # only for debugging, will pick first n number of samples deterministically
    ):
        self._processed_file_paths = None
        self._transform = transform
        self._pre_transform = pre_transform
        self._edge_repr = edge_repr
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

        ##### NOTE: only for debugging purposes ########
        self._num_samples_debug_mode = num_samples_debug_mode
        ##### NOTE: only for debugging purposes ########

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
        """Convert between string representation to multi hot encoding of correct node types."""
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
            results = []
            self.load_metadata()
            generation_steps = []
            future_saved_file_paths = []
            chunk_size = 1000
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_gen_steps_to_pkl_file_path = [
                    executor.submit(
                        self._convert_data_shard_to_list_of_trace_steps, pkl_file_path
                    )
                    for pkl_file_path in self.raw_file_names
                ]
                with tqdm(total=len(self.raw_file_names)) as pbar:
                    for future_gen_steps in concurrent.futures.as_completed(
                        future_gen_steps_to_pkl_file_path
                    ):
                        current_generation_steps = future_gen_steps.result()

                        # put all the generatoin steps into the queue
                        generation_steps += current_generation_steps
                        pbar.update(1)

                future_saved_file_paths += [
                    executor.submit(self._save_processed_gen_step, chunk, i)
                    for i, chunk in enumerate(
                        chunk_list(generation_steps, chunk_size=chunk_size)
                    )
                ]

                with tqdm(total=len(self.raw_file_names)) as pbar:
                    for future in concurrent.futures.as_completed(
                        future_saved_file_paths
                    ):
                        results += [future.result()]
                        pbar.update(1)

                # accumulated_generation_steps = []
                # with tqdm(total = len(self.raw_file_names)) as pbar:
                #     for future_gen_steps in concurrent.futures.as_completed(future_gen_steps_to_pkl_file_path):
                #         pkl_file_path = future_gen_steps_to_pkl_file_path[future_gen_steps]
                #         current_generation_steps = future_gen_steps.result()

                #         accumulated_generation_steps += current_generation_steps
                #         accumulated_num_steps += len(current_generation_steps)

                #         if accumulated_num_steps > 200:
                #             generation_steps.append((accumulated_generation_steps, pkl_file_path))
                #             accumulated_generation_steps = []
                #         pbar.update(1)

                # future_saved_file_paths = [executor.submit(self._save_processed_gen_step, molecule_gen_steps, pkl_file_path) for molecule_gen_steps, pkl_file_path in generation_steps]
                # with tqdm(total = len(generation_steps)) as pbar:
                #     for future in concurrent.futures.as_completed(future_saved_file_paths):
                #         results += [future.result()]
                #         pbar.update(1)

            self.generate_preprocessed_file_paths_csv(
                preprocessed_file_paths_folder=os.path.join(
                    self._output_pyg_trace_dataset_parent_folder, self._split
                ),
                results=results,
            )

    def _save_processed_gen_step(self, molecule_gen_steps, id):
        """Saves a list of trace steps corresponding to different molecules."""

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

        # file_name = (
        #     f'{pkl_file_path.split("/")[-1].split(".")[0]}_nsteps_{len(molecule_gen_steps)}.pkl.gz'  #
        # )
        file_name = f"{id}_nsteps_{len(molecule_gen_steps)}.pkl.gz"

        file_path = os.path.join(
            self._output_pyg_trace_dataset_parent_folder,
            self._split,
            file_name,
        )
        # batch them together
        molecule_gen_steps = Batch.from_data_list(
            molecule_gen_steps,
            follow_batch=[
                "correct_edge_choices",
                "correct_edge_types",
                "valid_edge_choices",
                "valid_attachment_point_choices",
                "correct_attachment_point_choice",
                "correct_node_type_choices",
                "original_graph_x",
                "correct_first_node_type_choices",
                # pick attachment points
                "candidate_attachment_points",
                # pick edge
                "candidate_edge_targets",
            ],
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
            for molecule in molecules:
                generation_steps += self._extract_generation_steps(molecule)

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
            if (
                self._using_self_loops
            ):  # by default this is not used since pyg layers cover this
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

            if self._edge_repr == EdgeRepresentation.edge_type:
                gen_step_features["original_graph_edge_features"] = np.array(edge_types)
            elif self._edge_repr == EdgeRepresentation.edge_attr:
                edge_attr = edge_types
                # TODO: add other edge features produced from preprocessing
                gen_step_features["original_graph_edge_features"] = np.array(edge_attr)
            else:
                raise NotImplementedError

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

            if self._edge_repr == EdgeRepresentation.edge_type:
                gen_step_features["partial_graph_edge_features"] = np.array(edge_types)
            elif self._edge_repr == EdgeRepresentation.edge_attr:
                edge_attr = edge_types
                # TODO: add other edge features produced from preprocessing
                gen_step_features["partial_graph_edge_features"] = np.array(edge_attr)
            else:
                raise NotImplementedError

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
                gen_step_features["correct_node_type_choices"] = np.array(
                    [self.node_types_to_multi_hot(gen_step.correct_node_type_choices)]
                )
            else:
                gen_step_features["correct_node_type_choices"] = np.zeros(
                    shape=(0,) + (self.num_node_types,)
                )
            if molecule.correct_first_node_type_choices is not None:
                gen_step_features["correct_first_node_type_choices"] = np.array(
                    [
                        self.node_types_to_multi_hot(
                            molecule.correct_first_node_type_choices
                        )
                    ]
                )
            else:
                gen_step_features["correct_first_node_type_choices"] = np.zeros(
                    shape=(0,) + (self.num_node_types,)
                )
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

        file_path = self.processed_file_names[idx]
        with gzip.open(file_path, "rb") as f:
            data = pickle.load(f)
        if self._num_samples_debug_mode is not None:
            # NOTE: only for debugging; deterministically pick first n samples
            # and return them instead of random subsampling
            unrolled = data.to_data_list()
            return Batch.from_data_list(
                unrolled[: self._num_samples_debug_mode],
                follow_batch=[
                    "correct_edge_choices",
                    "correct_edge_types",
                    "valid_edge_choices",
                    "valid_attachment_point_choices",
                    "correct_attachment_point_choice",
                    "correct_node_type_choices",
                    "original_graph_x",
                    "correct_first_node_type_choices",
                    # pick attachment points
                    "candidate_attachment_points",
                    # pick edge
                    "candidate_edge_targets",
                ],
            )
        if "train" in self._split and self._gen_step_drop_probability > 0:
            unrolled = data.to_data_list()
            selected_idx = np.arange(len(unrolled))[
                np.random.rand(len(unrolled)) > self._gen_step_drop_probability
            ]
            data = Batch.from_data_list(
                [unrolled[i] for i in selected_idx],
                follow_batch=[
                    "correct_edge_choices",
                    "correct_edge_types",
                    "valid_edge_choices",
                    "valid_attachment_point_choices",
                    "correct_attachment_point_choice",
                    "correct_node_type_choices",
                    "original_graph_x",
                    "correct_first_node_type_choices",
                    # pick attachment points
                    "candidate_attachment_points",
                    # pick edge
                    "candidate_edge_targets",
                ],
            )
            return data
        else:
            return data

        ###################################################################################
        # DEPRECATED: Previously we read in a list of individual trace steps, but now we batch them together
        # during `process()` itself
        # if self._current_in_memory_data_shard is not None:
        #     while not self._reached_end_of_current_data_shard():
        #         # decide whether to drop the current generation step just like in MoLeR

        #         if not self._drop_current_gen_step():
        #             data = self._current_in_memory_data_shard[
        #                 self._current_in_memory_data_shard_counter
        #             ]
        #             self._current_in_memory_data_shard_counter += 1
        #             return data

        #         self._current_in_memory_data_shard_counter += 1

        #     file_path = self.processed_file_names[idx]
        #     return self._read_in_and_extract_first_item_of_data_shard(file_path)
        # else:

        #     file_path = self.processed_file_names[idx]
        #     return self._read_in_and_extract_first_item_of_data_shard(file_path)
        ###################################################################################

        # alternative for reading in individual .pt files (NOTE currently infeasible)


def str_to_int(row):
    """Specifically for the L1000 csv"""
    row["ControlIndices"] = np.asarray(row["ControlIndices"].split(" "), dtype=np.int32)
    row["TumourIndices"] = np.asarray(row["TumourIndices"].split(" "), dtype=np.int32)
    return row


class LincsDataset(MolerDataset):
    def __init__(
        self,
        root,
        raw_moler_trace_dataset_parent_folder,  # absolute path
        output_pyg_trace_dataset_parent_folder,  # absolute path
        gene_exp_controls_file_path,
        gene_exp_tumour_file_path,
        lincs_csv_file_path,
        split="train",
        transform=None,
        pre_transform=None,
        using_self_loops=False,
        gen_step_drop_probability=0.5,
        edge_repr=EdgeRepresentation.edge_attr,
        num_samples_debug_mode=None,  # only for debugging, will pick first n number of samples deterministically
    ):
        super().__init__(
            root=root,
            raw_moler_trace_dataset_parent_folder=raw_moler_trace_dataset_parent_folder,  # absolute path
            output_pyg_trace_dataset_parent_folder=output_pyg_trace_dataset_parent_folder,  # absolute path
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            using_self_loops=using_self_loops,
            gen_step_drop_probability=gen_step_drop_probability,
            edge_repr=edge_repr,
            num_samples_debug_mode=num_samples_debug_mode,  # only for debugging, will pick first n number of samples deterministically
        )
        print("Loading controls gene expression...")
        self._gene_exp_controls = load(gene_exp_controls_file_path, allow_pickle=True)["genes"].astype('float32')
        print("Loading tumour gene expression...")
        self._gene_exp_tumour = load(gene_exp_tumour_file_path, allow_pickle=True)["genes"].astype('float32')
        print("Loading csv...")
        self._lincs_df = pd.read_csv(
            lincs_csv_file_path
        )  # expects the whole dataframe with train, validation and test splits
        self._lincs_df = self._lincs_df.apply(lambda x: str_to_int(x), axis=1)
        self._experiment_idx_to_control_gene_exp_idx = self._lincs_df[
            "ControlIndices"
        ].values  # numpy array
        self._experiment_idx_to_tumour_gene_exp_idx = self._lincs_df[
            "TumourIndices"
        ].values  # numpy array
        self._experiment_idx_to_dose = np.log1p(self._lincs_df['Dose'].values)/ np.log(11.) # logarithmic scale, s.t. 10 micromoles -> 1 unit
        del self._lincs_df

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
            if (
                self._using_self_loops
            ):  # by default this is not used since pyg layers cover this
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

            if self._edge_repr == EdgeRepresentation.edge_type:
                gen_step_features["original_graph_edge_features"] = np.array(edge_types)
            elif self._edge_repr == EdgeRepresentation.edge_attr:
                edge_attr = edge_types
                # TODO: add other edge features produced from preprocessing
                gen_step_features["original_graph_edge_features"] = np.array(edge_attr)
            else:
                raise NotImplementedError

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

            if self._edge_repr == EdgeRepresentation.edge_type:
                gen_step_features["partial_graph_edge_features"] = np.array(edge_types)
            elif self._edge_repr == EdgeRepresentation.edge_attr:
                edge_attr = edge_types
                # TODO: add other edge features produced from preprocessing
                gen_step_features["partial_graph_edge_features"] = np.array(edge_attr)
            else:
                raise NotImplementedError

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
                gen_step_features["correct_node_type_choices"] = np.array(
                    [self.node_types_to_multi_hot(gen_step.correct_node_type_choices)]
                )
            else:
                gen_step_features["correct_node_type_choices"] = np.zeros(
                    shape=(0,) + (self.num_node_types,)
                )
            if molecule.correct_first_node_type_choices is not None:
                gen_step_features["correct_first_node_type_choices"] = np.array(
                    [
                        self.node_types_to_multi_hot(
                            molecule.correct_first_node_type_choices
                        )
                    ]
                )
            else:
                gen_step_features["correct_first_node_type_choices"] = np.zeros(
                    shape=(0,) + (self.num_node_types,)
                )
            # Add graph_property_values
            gen_step_features = {**gen_step_features, **molecule_property_values}
            gen_step_features["l1000_idx"] = gen_step.idx
            molecule_gen_steps += [gen_step_features]

        molecule_gen_steps = self._to_tensor_moler(molecule_gen_steps)

        return [MolerData(**step) for step in molecule_gen_steps]

    def get(self, idx):
        """
        This is a workaround for reading in one data shard at a time (one molecule at a time)
        Each molecule has a varying number of generation steps, and all the generation steps
        for one particular molecule will be stored in a single data shard.
        We use the idx to reference the molecule idx and read in each data shard and store it
        as an attribute in the class. Then, we maintain a counter for iterating through the
        shard. Once we read the end of the data shard, we use the idx to read in another molecule.
        """

        file_path = self.processed_file_names[idx]
        with gzip.open(file_path, "rb") as f:
            data = pickle.load(f)
        if self._num_samples_debug_mode is not None:
            # NOTE: only for debugging; deterministically pick first n samples
            # and return them instead of random subsampling
            unrolled = data.to_data_list()
            return Batch.from_data_list(
                unrolled[: self._num_samples_debug_mode],
                follow_batch=[
                    "correct_edge_choices",
                    "correct_edge_types",
                    "valid_edge_choices",
                    "valid_attachment_point_choices",
                    "correct_attachment_point_choice",
                    "correct_node_type_choices",
                    "original_graph_x",
                    "correct_first_node_type_choices",
                    # pick attachment points
                    "candidate_attachment_points",
                    # pick edge
                    "candidate_edge_targets",
                ],
            )

        if  "train" in self._split and self._gen_step_drop_probability > 0:
            unrolled = data.to_data_list()
            selected_idx = np.arange(len(unrolled))[
                np.random.rand(len(unrolled)) > self._gen_step_drop_probability
            ]
            data = Batch.from_data_list(
                [unrolled[i] for i in selected_idx],
                follow_batch=[
                    "correct_edge_choices",
                    "correct_edge_types",
                    "valid_edge_choices",
                    "valid_attachment_point_choices",
                    "correct_attachment_point_choice",
                    "correct_node_type_choices",
                    "original_graph_x",
                    "correct_first_node_type_choices",
                    # pick attachment points
                    "candidate_attachment_points",
                    # pick edge
                    "candidate_edge_targets",
                ],
            )
        # given the df row idx, we want to index into the df, then get the control
        # and tumour indices => then we randomly pick one from each and index into the
        # torch tensor.

        # Method 1: directly find row from df using iloc => likely slower because
        # it is O(n) but likely less space complexity because each training, and validation
        # dataset will only need to store its own split of the df

        # Method 2: store the entire df and the entire npz for controls and tumous gene expressions
        # this will be faster because we can just directly pick from the npz
        # similar to in  https://github.com/insilicomedicine/BiAAE/blob/master/dataloader/lincs_dl.py

        # uses method 2
        experiment_idx = data.l1000_idx  # get row idx
        control_idx = self._experiment_idx_to_control_gene_exp_idx[
            experiment_idx
        ]  # get list of control idx
        gene_exp_control_idx = [
            random.randint(0, len(arr)) for arr in control_idx
        ]  # choose one of the control expt idx for each sample in the batch
        tumour_idx = self._experiment_idx_to_tumour_gene_exp_idx[experiment_idx]
        gene_exp_tumour_idx = [random.randint(0, len(arr)) for arr in tumour_idx]
        control_gene_exp = self._gene_exp_controls[
            gene_exp_control_idx
        ]  # torch tensors of size batch_size x gene exp dim
        tumour_gene_exp = self._gene_exp_tumour[gene_exp_tumour_idx]
        diff_gene_exp = tumour_gene_exp - control_gene_exp
        data.gene_expressions = torch.from_numpy(diff_gene_exp).float()
        data.dose = torch.from_numpy(self._experiment_idx_to_dose[experiment_idx])
        return data
