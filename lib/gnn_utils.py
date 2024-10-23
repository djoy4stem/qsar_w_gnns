import os
from os.path import join
import sys

from typing import List, Union, Any, Set, Dict
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Data
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
        if add_explicit_h:
            molecule = Chem.AddHs(molecule)
        mol_features = None
        if compute_global_features:
            mol_features = mol_featurizer.compute_properties_for_mols(
                [molecule], as_dataframe=False
            )

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
                add_global_feat_to_nodes=False,
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
                    if not add_global_feat_to_nodes:
                        # print(mol_features[i].values())
                        data.global_feats = torch.as_tensor(mf, dtype=torch.float)
                    else:
                        # print(data.x, data.x.shape)
                        merged = torch.cat(
                            (data.x, torch.as_tensor([mf] * data.x.shape[0])), dim=1
                        )
                        data.x = merged
                        # print(data.x)
                        # print(data.x.shape)

                    graphs.append(data)
        # print("Graph from mol 1", f" x {graphs[0].x.shape} - edge_attr {graphs[0].edge_attr.shape}")
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
):
    graph_dataset = graph_from_smiles_list(
        smiles_list=smiles_list,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        add_explicit_h=add_explicit_h,
        mol_featurizer=mol_featurizer,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
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
):
    graph_dataset = graphs_from_mol_list(
        molecules=mol_list,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        add_explicit_h=add_explicit_h,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
    )

    ## Add targets
    # print(f'len(graph_dataset) = {len(graph_dataset)}')
    for i in range(len(graph_dataset)):
        graph_dataset[i].y = torch.as_tensor(
            [targets[i]]
        )  # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]
        # print(f'{i}', graph_dataset[i].global_feats.shape, graph_dataset[i].global_feats)

    return graph_dataset
