We need protobuf<3.20, but protobuf<3.20 gives an error when importing tf
> ImportError: cannot import name 'builder' from 'google.protobuf.internal'

To fix the problem: 
1. upgrade protobuf to the latest version
2. copy its `builder.py` file from `~/miniconda3/envs/druggen/lib/python3.9/site-packages/google/protobuf/internal`
3. downgrade protobuf to 3.19.6, and paste the file back to the corresponding folder

Reference: https://stackoverflow.com/a/72494013