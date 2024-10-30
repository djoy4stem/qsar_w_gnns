import os
from os.path import join
import sys
import copy

from typing import List, Union, Any, Set, Dict
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import networkx.algorithms.isomorphism as iso
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from IPython.display import display

from joblib import Parallel, delayed

from lib import utilities
from lib import featurizers as feat
from lib.featurizers import AtomFeaturizer, BondFeaturizer


RDLogger.DisableLog("rdApp.*")


def graph_from_molecule(
    molecule: AllChem.Mol,
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    add_explicit_h: bool=False,
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
):
    """ """

    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    mol_features = None

    assert not (
        (not compute_global_features) and add_global_feat_to_nodes
    ), "If compute_global_features==False, then add_global_feat_to_nodes cannot be True."

    # try:
    if True:

        mol_features = None
        if compute_global_features:
            mol_features = mol_featurizer.compute_properties_for_mols(
                [molecule], as_dataframe=False, add_explicit_h=True
            )

        if not add_explicit_h:
            molecule = Chem.RemoveHs(molecule)

        for atom in molecule.GetAtoms():
            atom_features.append(atom_featurizer.encode(atom))

            # Add self-loops
            pair_indices.append([atom.GetIdx(), atom.GetIdx()])
            bond_features.append(bond_featurizer.encode(None))

            for neighbor in atom.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
                bond_features.append(bond_featurizer.encode(bond))

        if add_global_feat_to_nodes:
            atom_features = [
                np.array(af.tolist() + mol_features) for af in atom_features
            ]
            # mol_features  = None
            # print("After adding mol. feats.: ", atom_features[0][-15:])

        ## using as_tensor (or from_numpy) converts the numpy array to a tensor WITHOUT COPYING the data
        atom_features = torch.as_tensor(
            np.array(atom_features), dtype=torch.float
        ).view(-1, len(atom_features[0]))
        bond_features = torch.as_tensor(np.array(bond_features), dtype=torch.long).view(
            -1, len(bond_features[0])
        )
        pair_indices = torch.as_tensor(pair_indices).t().to(torch.long).view(2, -1)

        if not mol_features is None:
            # print("===>", mol_features)
            mol_features = torch.as_tensor(
                mol_features, dtype=torch.float
            )  # .unsqueeze(0)
            # print(f"mol_features= {mol_features.shape} :: {mol_features}")

        # return np.array(atom_features), np.array(bond_features), np.array(pair_indices)
        return Data(
            x=atom_features,
            edge_index=pair_indices,
            edge_attr=bond_features,
            global_feats=mol_features,
        )
    # except Exception as exp:
    #     print("Could not compute graph from molecule.")
    #     return None


def graphs_from_mol_list(
    molecules: [AllChem.Mol],
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    add_explicit_h: bool = False,
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = False,
    feature_scaler: bool = None 
):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    mol_features = None

    assert not (
        (not compute_global_features) and add_global_feat_to_nodes
    ), "If compute_global_features==False, then add_global_feat_to_nodes cannot be True."

    # try:
    if True:
        if add_explicit_h:
            molecules = [Chem.AddHs(mol) for mol in molecules]
        mol_features = None
        len_mol_features, len_ab_feats = 0, 0
        if compute_global_features:
            mol_features = mol_featurizer.compute_properties_for_mols(
                molecules, as_dataframe=False
            )
            len_mol_features = len(mol_features)

        print(f"len_mol_features: {len_mol_features}")
        # print(pd.DataFrame(molecules, columns=['RMol']))
        d = pd.DataFrame(molecules, columns=["RMol"])
        atom_and_bond_features_df = d["RMol"].apply(
            lambda mol: graph_from_molecule(
                mol,
                atom_featurizer=atom_featurizer,
                bond_featurizer=bond_featurizer,
                mol_featurizer=mol_featurizer,
                compute_global_features=False,
                add_global_feat_to_nodes=False
            )
        )  # .values.tolist()
        # len_ab_feats = atom_and_bond_features_df)
        len_ab_feats = atom_and_bond_features_df.shape[0]
        # print(f"len_mol_features = {len_mol_features}  /  atom_and_bond_features_df = {len_ab_feats}")
        # print(mol_features)
        # print(atom_and_bond_features_df)

        if (not mol_features is None) and len_mol_features != len_ab_feats:
            raise Exception(
                "Different numbers of items for atom/bond features and molecular features."
            )
        else:
            if len_mol_features == 0:
                return atom_and_bond_features_df.values.tolist()
            else:
                graphs = []
                for i in range(len_mol_features):
                    data = atom_and_bond_features_df.iloc[i]
                    mf = list(mol_features[i].values())
                    data.global_feats = torch.as_tensor(mf, dtype=torch.float)
                    # if not add_global_feat_to_nodes:
                    #     # print(mol_features[i].values())
                    #     data.global_feats = torch.as_tensor(mf, dtype=torch.float)
                    # else:
                    #     # print(data.x, data.x.shape)
                    #     merged = torch.cat(
                    #         (data.x, torch.as_tensor([mf] * data.x.shape[0])), dim=1
                    #     )
                    #     data.x = merged
                    #     # print(data.x)
                    #     # print(data.x.shape)
                    # print(f"data = {data}")
                    graphs.append(data)
        # print("Graph from mol 1", f" x {graphs[0].x.shape} - edge_attr {graphs[0].edge_attr.shape}")


        print("Clean features...")
        graphs = clean_features_from_data_and_batch(data=graphs,
                    scale_features = scale_features,
                    feature_scaler=feature_scaler,
                    add_global_feats_to_nodes=add_global_feat_to_nodes,
                    return_as_list = True
)


        return graphs

    # except Exception as exp:
    #     print("Could not compute graphs from molecules.")
    #     return None


def graph_from_smiles_list(
    smiles_list: List[str],
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    add_explicit_h: bool = True,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = False,
    feature_scaler: bool = None 
):
    molecules = [
        utilities.molecule_from_smiles(smiles, add_explicit_h) for smiles in smiles_list
    ]
    graphs = graphs_from_mol_list(
        molecules=molecules,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
        scale_features=scale_features,
        feature_scaler=feature_scaler
    )

    return graphs


def get_dataset_from_smiles_list(
    smiles_list: List[str],
    targets: List[Any],
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    add_explicit_h: bool = True,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = False,
    feature_scaler: bool = None    
):
    graph_dataset = graph_from_smiles_list(
        smiles_list=smiles_list,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        add_explicit_h=add_explicit_h,
        mol_featurizer=mol_featurizer,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
        scale_features=scale_features,
        feature_scaler=feature_scaler
    )

    ## Add targets
    for i in range(len(targets)):
        graph_dataset[i].y = torch.as_tensor(
            [targets[i]]
        )  # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]

    return graph_dataset


def get_dataset_from_dframe(
    input_df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    add_explicit_h: bool = True,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = False,
    feature_scaler: bool = None
):
    # print("input_df.shape=", input_df.shape)
    # print(input_df
    graph_dataset = graph_from_smiles_list(
        smiles_list=input_df[smiles_column],
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        add_explicit_h=add_explicit_h,
        mol_featurizer=mol_featurizer,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
        scale_features=scale_features,
        feature_scaler=feature_scaler
    )

    # print("graph_dataset = ", graph_dataset)
    # print(len(graph_dataset))
    targets = input_df[target_column].values
    # print("targets = ", targets)

    ## Add targets
    for i in range(len(targets)):
        graph_dataset[i].y = torch.as_tensor(
            [targets[i]]
        )  # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]

    return graph_dataset


def get_dataset_from_mol_list(
    mol_list: List[AllChem.Mol],
    targets: List[Any],
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    add_explicit_h: bool = True,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = False,
    feature_scaler: bool = None
):
    graph_dataset = graphs_from_mol_list(
        molecules=mol_list,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        add_explicit_h=add_explicit_h,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
        scale_features=scale_features,
        feature_scaler=feature_scaler
    )

    ## Add targets
    # print(f'len(graph_dataset) = {len(graph_dataset)}')
    for i in range(len(graph_dataset)):
        graph_dataset[i].y = torch.as_tensor(
            [targets[i]]
        )  # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]
        # print(f'{i}', graph_dataset[i].global_feats.shape, graph_dataset[i].global_feats)

    return graph_dataset



def add_global_features_to_nodes(data: Union[Data, Batch]):
    # def cat_1d_to_2d_tensor(tensor_1d, tensor_2d):
    #     # expland tensor_1d (shape = d) using tensor_2d (shape = [n,m])
    #     tensor_1d_expanded = tensor_1d.unsqueeze(0).expand(tensor_2d.size(0), -1)  # Shape: [n, d]
    #     # Concatenate along the feature dimension (columns)
    #     t_concat = torch.cat([tensor_2d, tensor_1d_expanded], dim=1)  # Shape: [n, m + d]
    #     return t_concat
    new_data = copy.deepcopy(data)
    if isinstance(data, List):
        for i in range(len(data)):
            # print(data[i].global_feats)
            new_data[i].x = utilities.concat_1d_to_2d_tensor(torch.tensor(new_data[i].global_feats), new_data[i].x)
            new_data[i].global_feats = None
        
    elif isinstance(data, Batch):
        new_data = new_data.to_data_list()
        for i in range(len(data)):
            # print(data[i].global_feats)
            new_data[i].x = utilities.concat_1d_to_2d_tensor(tensor(new_data[i].global_feats), new_data[i].x)
            new_data[i].global_feats = None

        new_data = Batch.from_data_list(new_data)
                

    return new_data


def clean_features_from_data_and_batch(data: Union[Batch, List[Data]],
                        scale_features: bool = True,
                        feature_scaler=MinMaxScaler(),
                        add_global_feats_to_nodes: bool = False,
                        return_as_list: bool = False
):

    has_global_feats = False
    num_data_objects, dim_global_feats = None, None 
    global_feats_reshaped = None
    cleaned_data = data

        

    ## does that have globale features
    if isinstance(data, List) and len(data)>0:
        assert isinstance(data[0], Data), "Error: All elements must be of type Data."
        # print(f"First class: {data[0].__class__}")
        # for i in range(len(data)):
            # # print(data[i])
            # if hasattr(data[i], "global_feats") and data[i].global_feats is None:
            #     print(f"{i} has None global_feats")
            # if not hasattr(data[i], "global_feats"):
            #     print(f"{i} has no global_feats")

        all_have_global_feats = all([hasattr(d, "global_feats")  and not d.global_feats is None for d in data])
        assert all_have_global_feats, "Error: Some Data objects of the list either have a non-existing or null attribute global_feats."

        has_global_feats = all_have_global_feats

        num_data_objects = len(data)
        dim_global_feats = data[0].global_feats.shape[-1]

        global_feats_reshaped = [d.global_feats for d in data]


    elif isinstance(data, batch):
        has_global_feats = (
            hasattr(data, "global_feats") and not data.global_feats is None
        )

        # Reshape global_feats to a 2D tensor
        num_data_objects = data.batch.max().item() + 1
        dim_global_feats = int(data.global_feats.shape[0]/num_data_objects) ## This must be an integer
        global_feats_reshaped = data.global_feats.view(num_data_objects, dim_global_feats)

    if scale_features:
        assert (not feature_scaler is None), f"ValueError: Provide a valid feature scaler. None provided."
        # Clean/Scale
        global_feats_reshaped = utilities.clean_features(
            features=global_feats_reshaped, feature_scaler=feature_scaler
        )
        print("global_feats_reshaped", global_feats_reshaped.shape)
        # print(global_feats_reshaped.__class__)

        if isinstance(data, List):
            cleaned_data = data
            for i in range(len(data)):
                cleaned_data[i].global_feats = global_feats_reshaped[i]

        elif isinstance(data, batch):
            # Flatten and assign back
            print("tensor(global_feats_reshaped).view(-1)", tensor(global_feats_reshaped).view(-1))
            cleaned_data.global_feats = tensor(global_feats_reshaped).view(-1)
    
    if add_global_feats_to_nodes:
        if not has_global_feats:
            warnings.warn("There are no global features to add. The data will be returned as is.")
        else:
            cleaned_data = add_global_features_to_nodes(cleaned_data)


    return cleaned_data