from dataset import MolerDataset

# for 398 using previous implementation where multithreading is only applied to each molecule, we get
# 2:08:48 hours and 
valid_dataset = MolerDataset(
    root="/data/ongh0068",
    raw_moler_trace_dataset_parent_folder='/data/ongh0068/guacamol/trace_dir',#"/data/ongh0068/l1000/trace_playground",
    output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/test_thread",
    split="valid_0",
)

# train_dataset = MolerDataset(
#     root="/data/ongh0068",
#     raw_moler_trace_dataset_parent_folder='/data/ongh0068/guacamol/trace_dir',#"/data/ongh0068/l1000/trace_playground",
#     output_pyg_trace_dataset_parent_folder="/data/ongh0068/l1000/pyg_output_playground",
#     split="train_0",
# )

