from dataset import MolerDataset, LincsDataset
""" molecule_generation preprocess l1000_biaae/INPUT_DIR l1000_biaae/OUTPUT_DIR l1000_biaae/TRACE_DIR --motif_vocabulary_provided='/data/ongh0068/guacamol/trace_dir/metadata.pkl.gz' --using_lincs"""
valid_dataset = LincsDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/TRACE_DIR",  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/already_batched",
    gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
    gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
    lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
    split="valid_0",
)

train_dataset = LincsDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/TRACE_DIR",  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/already_batched",
    gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
    gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
    lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
    split="train_0",
)
test_dataset = LincsDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/TRACE_DIR",  # "/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/l1000_biaae/already_batched",
    gene_exp_controls_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_controls.npz",
    gene_exp_tumour_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/robust_normalized_tumors.npz",
    lincs_csv_file_path="/data/ongh0068/l1000/l1000_biaae/lincs/experiments_filtered.csv",
    split="test_0",
)
# for 398 using previous implementation where multithreading is only applied to each molecule, we get
# 2:08:48 hours and 2.7 GB


# Then with batching multiple molecules in a list, we get the same time and 2.0 GB space


# Now with actually batching the pyg objects themselves (each with > 5000 trace steps), we get 1.4GB

# same disk space with > 1000 steps
# valid_dataset = MolerDataset(
#     root="/data/ongh0068",
#     raw_moler_trace_dataset_parent_folder='/data/ongh0068/guacamol/trace_dir',#"/data/ongh0068/l1000/trace_playground",
#     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
#     split="valid_0",
# )

# train_dataset = MolerDataset(
#     root="/data/ongh0068",
#     raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",  # "/data/ongh0068/l1000/trace_playground",
#     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
#     split="train_4000",
# )

# train_dataset = MolerDataset(
#     root="/data/ongh0068",
#     raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",  # "/data/ongh0068/l1000/trace_playground",
#     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
#     split="train_5000",
# )

# train_dataset = MolerDataset(
#     root="/data/ongh0068",
#     raw_moler_trace_dataset_parent_folder="/data/ongh0068/guacamol/trace_dir",  # "/data/ongh0068/l1000/trace_playground",
#     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/already_batched",
#     split="train_6000",
# )
