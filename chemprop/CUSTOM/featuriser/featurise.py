import os
import sys

import pandas as pd
from chemprop import data, featurizers
import json
import numpy as np
# 1. Parse the sdf files in the directory 'data/processed/sdf_files'
# 2. In each sdf file, there are three molecules - 'r1h', 'r2h', 'ts' - we are interested in the 'r1h', 'r2h' molecules
# 3. During the data.Moleculepoint loop, we will need to extract 'r1h' and 'r2h' molecules from the sdf file - set as mol_type='r1h' and mol_type for each molecule

MOL_TYPES = ['r1h', 'r2h']

def read_target_data(path='data/processed/target_data.csv', set_col_index=True, column_targets=None):
    """
    Read the target data from the csv file 'data/processed/target_data.csv'
    
    Returns:
        pd.DataFrame: Target data
    """
    if set_col_index:
        target_data = pd.read_csv(path, index_col=0)
    else:
        target_data = pd.read_csv(path)

    if column_targets is not None:
        # Check if the columns exist in the dataframe
        missing_columns = [col for col in column_targets if col not in target_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in target data: {missing_columns}")
        
        # Filter the dataframe to only include the specified columns
        target_data = target_data[column_targets]

    return target_data

def get_sdf_files(path='data/processed/sdf_files'):
    """
    Get the sdf files, including path,  in the directory 'data/processed/sdf_files'
    
    Returns:
        list: List of sdf files
    """

    sdf_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.sdf')]
    
    return sdf_files

def extract_rxn_from_path(sdf_path):
    """
    Extract the reaction name from the sdf file path. 
    
    For example, '/path/to/ABC123.sdf' yields 'ABC123'.
    
    Parameters:
        sdf_path (str): Path to the sdf file
        
    Returns:
        str: Reaction name
    """
    base = os.path.basename(sdf_path)
    rxn_id, _ = os.path.splitext(base)

    return rxn_id

def load_datapoints(sdf_paths, mol_types, keep_h=True, add_h=False, sanitize: bool = False, target_data: pd.DataFrame=None, include_extra_features=False):
    """
    Load datapoints from a list of SDF files, extracting specified molecule types.

    Parameters
    ----------
    sdf_paths : list of str
        List of file paths to the SDF files.
    mol_types : list of str
        List of molecule types (e.g., ["r1h", "r2h"]) to extract from each SDF file.
    keep_h : bool, optional
        Whether to keep hydrogens in the molecule, by default True.
    add_h : bool, optional
        Whether to add hydrogens to each molecule, by default True.
    target_data : pd.DataFrame, optional
        Target data to assign to each datapoint, by default None.
    include_extra_features : bool, optional
        Whether to include extra features in the datapoints, by default False.

    Returns
    -------
    list of list
        A list where each inner list contains MoleculeDatapoint objects for a given molecule type.
        For example, the first inner list holds datapoints for mol_types[0] from all SDF files,
        the
    """

    target_data_dict = target_data.to_dict('index')

    # Number of components
    num_components = len(mol_types)
    all_data = [[] for _ in range(num_components)]

    for sdf_path in sdf_paths:
        rxn_id = extract_rxn_from_path(sdf_path)
        if rxn_id not in target_data_dict:
            print(f"Reaction {rxn_id} not found in target data")
            continue
        target = list(target_data_dict[rxn_id].values())
        # Convert all values to float
        # find which value is not a float and print it and its type
        for val in target:
            if not isinstance(val, float):
                print(f"Value: {val}, Type: {type(val)}")


        target = [float(val) for val in target]
        target = np.array(target, dtype=np.float32)
        dp0 = data.MoleculeDatapoint.from_sdf(sdf=sdf_path, mol_type=mol_types[0], keep_h=keep_h, add_h=add_h, sanitize=sanitize, include_extra_features=include_extra_features, y=target)

        all_data[0].append(dp0)

        for comp_index in range(1, num_components):
            dp = data.MoleculeDatapoint.from_sdf(sdf=sdf_path, mol_type=mol_types[comp_index], keep_h=keep_h, sanitize=sanitize, include_extra_features = include_extra_features, add_h=add_h)
            all_data[comp_index].append(dp)
    
    return all_data

def featurise_datapoints(datapoints, featurizer):
    """
    Create a MoleculeDataset from a list of datapoints and a featurizer.

    Parameters
    ----------
    datapoints : list
        A list of MoleculeDatapoint objects.
    featurizer : object
        A featurizer instance (e.g., SimpleMoleculeMolGraphFeaturizer)
        that will be used to featurize the molecules.

    Returns
    -------
    MoleculeDataset
        An instance of MoleculeDataset with featurized datapoints.
    """
    return data.MoleculeDataset(datapoints, featurizer)


def build_extra_features(mol):
    """
    Build an extra feature matrix for the molecule using its mol_properties and electro_map.
    Assumes mol_properties and electro_map are stored as JSON strings on the molecule.
    
    Returns:
        np.ndarray of shape (n_atoms, 7)
    """
    n_atoms = mol.GetNumAtoms()
    extra_dim = 7  # 4 from mol_properties, 3 from electro_map
    features = np.zeros((n_atoms, extra_dim), dtype=np.float32)
    
    # Parse molecule-level properties, if they exist.
    try:
        mol_props = json.loads(mol.GetProp('mol_properties')) if mol.HasProp('mol_properties') else {}
    except Exception:
        mol_props = {}
    try:
        elec_map = json.loads(mol.GetProp('electro_map')) if mol.HasProp('electro_map') else {}
    except Exception:
        elec_map = {}
    
    for i in range(n_atoms):
        idx = str(i)
        # Process mol_properties: assign flags to 4 slots.
        # For example, we might decide on:
        #   slot 0: d_hydrogen, slot 1: a_hydrogen, slot 2: donator, slot 3: acceptor.
        if idx in mol_props:
            label = mol_props[idx].get("label", "").lower()
            if label == "d_hydrogen":
                features[i, 0] = 1.0
            elif label == "a_hydrogen":
                features[i, 1] = 1.0
            elif label == "donator":
                features[i, 2] = 1.0
            elif label == "acceptor":
                features[i, 3] = 1.0
        
        # Process electro_map: assign the R, A, D values to slots 4, 5, 6.
        if idx in elec_map:
            try:
                r_val = elec_map[idx].get("R")
                a_val = elec_map[idx].get("A")
                d_val = elec_map[idx].get("D")
                # Use np.nan as a sentinel if the value is missing or "None"
                features[i, 4] = float(r_val) if r_val not in [None, "None"] else np.nan
                features[i, 5] = float(a_val) if a_val not in [None, "None"] else np.nan
                features[i, 6] = float(d_val) if d_val not in [None, "None"] else np.nan
            except Exception:
                features[i, 4:7] = np.nan
        else:
            # If no electro_map info, assign nan to those slots.
            features[i, 4:7] = np.nan
            
    return features


def Featuriser(sdf_path, path, sanitize, include_extra_features=False, column_targets=None):
    sdf_files = get_sdf_files(sdf_path)
    target_data = read_target_data(path, set_col_index=True, column_targets=column_targets)
    datapoints = load_datapoints(sdf_files, MOL_TYPES, sanitize=sanitize, target_data=target_data, include_extra_features=include_extra_features)
    #featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    #dataset = [featurise_datapoints(datapoints[i], featurizer) for i in range(len(MOL_TYPES)) ]

    return datapoints

if __name__ == "__main__":
    dataset = Featuriser('/home/calvin/code/HAbstractionNet/data/processed/sdf_files')
