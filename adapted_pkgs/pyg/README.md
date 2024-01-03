# Custom mini-batching
The following changes are made to enable custom batching of the molecular generation steps during preprocessing in `~/miniconda3/envs/druggen/lib/python3.9/site-packages/torch_geometric/data/collate.py`. 
From line 80 and 81:

    if attr == 'ptr':
        continue

to line 80-86:

    if attr is not None:
        if 'ptr' in attr:
            tmp = values[0]
            for value in values[1:]:
                tmp = torch.cat((tmp, value[1:] + tmp[-1]))
            out_store[attr] = tmp
            continue

"ptr" attributes record the start and end node indices of each individual graph in a batch

# Custom pre-batching
In `~/miniconda3/envs/druggen/lib/python3.9/site-packages/torch_geometric/data/data.py`, append the following codes to line 512 and 513

    if value.size(0) == 0:
        return 0

This is to accommodate the special Moler generation step attribute 
**candidate attachment points**
which are the list of possible attachment points in the current subgraph. This list can be empty and this case is neglected by original PyG. The change is to fix the potential problem. 