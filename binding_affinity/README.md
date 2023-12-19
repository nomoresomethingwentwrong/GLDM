# Molecular docking experiments
Original perturbagens are given in SMILES format, and generated molecules are represented in graph structures. To perform docking, we need to generate 3D conformers for them. 

## Excape experiments

Excape dataset records protein targets of perturbagens. Intersect the smiles in l1000 and excape to filter for mols with interested protein targets. 
> only know which protein, have no idea about binding site structure. 

### Use the default conformer given by RDKit
Results are under `default_conformer`.
- `ligand`: ligand structures in sdf
    - `all_generated_mols`: all of the mols (356*100) produced in constrained generation experiments are stored
    - `original`: reference perturbagens
    - other folders: only the mols corresponding to the selected reference perturbagens, which has the highest binding affinity with the 9 selected target proteins
- `logs`: Gnina logs
    - `selected`: only the docking logs of selected mols. Docking is limited to the pocket (?) defined by reference perturbagen ligand
    - `selected_whole_pr`: only the docking logs of selected mols. Whole protein docking is performed 
    - other folders: docking logs of all generated mols. (**unfinished**)
- `poses`: Binding gestures. Content is *same with logs*

### Test all isomers
According to chemists, we should test docking of all isomers, and use the highest binding affinity scores of known perturbagens and generated mols to compare.
- `ligand`: only selected mols and original perturbagens
- `logs` & `poses`: results of selected 

## Binding DB experiments

> Unlike excape, bindingdb can filter interactions (between pr and small mol) with known crystal structures