def pprint_pyg_obj(batch):
    """For printing out the pytorch geometric attributes in a readable way."""
    for key in vars(batch)["_store"].keys():
        if key.startswith("_"):
            continue
        print(f"{key}: {batch[key].shape}")
