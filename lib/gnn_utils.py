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
from lib.featurizers import (
    AtomFeaturizer,
    BondFeaturizer,
    ATOM_FEATURIZER,
    BOND_FEATURIZER,
)


RDLogger.DisableLog("rdApp.*")


def graph_from_molecule(
    molecule: AllChem.Mol,
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    add_explicit_h: bool = False,
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
            mol_features = list(
                mol_featurizer.compute_properties_for_mols(
                    [molecule], as_dataframe=False
                )[0].values()
            )

        # print("mol_features", mol_features)
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
            # print("mol_features: ", mol_features)
            if isinstance(mol_features[0], (list, int, float)):
                atom_features = [
                    np.array(af.tolist() + [mol_features[0]]) for af in atom_features
                ]
            else:
                atom_features = [
                    np.array(af.tolist() + mol_features[0]) for af in atom_features
                ]   
            mol_features = None
        else:
            # print("mol_features", mol_features)
            if not mol_features is None:
                mol_features = torch.as_tensor(
                    mol_features, dtype=torch.float
                ).unsqueeze(0)

        ## using as_tensor (or from_numpy) converts the numpy array to a tensor WITHOUT COPYING the data
        atom_features = torch.as_tensor(
            np.array(atom_features), dtype=torch.float
        ).view(-1, len(atom_features[0]))
        bond_features = torch.as_tensor(np.array(bond_features), dtype=torch.long).view(
            -1, len(bond_features[0])
        )
        pair_indices = torch.as_tensor(pair_indices).t().to(torch.long).view(2, -1)
        # print("mol_features 2 = ", mol_features)

        # if not mol_features is None:
        #     print("===>", mol_features)
        #     mol_features = torch.as_tensor(
        #         list(mol_features.values()), dtype=torch.float
        #     )  # .unsqueeze(0)
        #     # print(f"mol_features= {mol_features.shape} :: {mol_features}")

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
    feature_scaler: bool = None,
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

        # print(f"len_mol_features: {len_mol_features}")
        # print(pd.DataFrame(molecules, columns=['RMol']))
        d = pd.DataFrame(molecules, columns=["RMol"])
        atom_and_bond_features_df = d["RMol"].apply(
            lambda mol: graph_from_molecule(
                mol,
                atom_featurizer=atom_featurizer,
                bond_featurizer=bond_featurizer,
                mol_featurizer=mol_featurizer,
                compute_global_features=compute_global_features,
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
                    data.global_feats = torch.as_tensor(mf, dtype=torch.float)
                    # print(data.global_feats)
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
        graphs = clean_features_from_data_and_batch(
            data=graphs,
            scale_features=scale_features,
            feature_scaler=feature_scaler,
            add_global_feats_to_nodes=add_global_feat_to_nodes,
            return_as_list=True,
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
    feature_scaler: bool = None,
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
        feature_scaler=feature_scaler,
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
    feature_scaler: bool = None,
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
        feature_scaler=feature_scaler,
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
    feature_scaler: bool = None,
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
        feature_scaler=feature_scaler,
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
    feature_scaler: bool = None,
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
        feature_scaler=feature_scaler,
    )
    # print("graph_dataset", graph_dataset)
    ## Add targets
    # print(f'len(graph_dataset) = {len(graph_dataset)}')
    for i in range(len(graph_dataset)):
        graph_dataset[i].y = torch.as_tensor(
            [targets[i]]
        )  # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]
        # print(f'{i}', graph_dataset[i].global_feats.shape, graph_dataset[i].global_feats)

    return graph_dataset


def add_global_features_to_nodes(data: Union[Data, Batch]):
    new_data = copy.deepcopy(data)
    if isinstance(data, List):
        for i in range(len(data)):
            # print(data[i].global_feats)
            new_data[i].x = utilities.concat_1d_to_2d_tensor(
                torch.tensor(new_data[i].global_feats), new_data[i].x
            )
            new_data[i].global_feats = None

    elif isinstance(data, Batch):
        new_data = new_data.to_data_list()
        for i in range(len(data)):
            # print(data[i].global_feats)
            new_data[i].x = utilities.concat_1d_to_2d_tensor(
                tensor(new_data[i].global_feats), new_data[i].x
            )
            new_data[i].global_feats = None

        new_data = Batch.from_data_list(new_data)

    return new_data


def clean_features_from_data_and_batch(
    data: Union[Batch, List[Data]],
    scale_features: bool = True,
    feature_scaler=MinMaxScaler(),
    add_global_feats_to_nodes: bool = False,
    return_as_list: bool = False,
):
    has_global_feats = False
    num_data_objects, dim_global_feats = None, None
    global_feats_reshaped = None
    cleaned_data = data

    ## does that have global features
    if isinstance(data, List) and len(data) > 0:
        assert isinstance(data[0], Data), "Error: All elements must be of type Data."

        all_have_global_feats = all(
            [hasattr(d, "global_feats") and not d.global_feats is None for d in data]
        )
        assert (
            all_have_global_feats
        ), "Error: Some Data objects of the list either have a non-existing or null attribute global_feats."

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
        dim_global_feats = int(
            data.global_feats.shape[0] / num_data_objects
        )  ## This must be an integer
        global_feats_reshaped = data.global_feats.view(
            num_data_objects, dim_global_feats
        )

    if scale_features:
        assert (
            not feature_scaler is None
        ), f"ValueError: Provide a valid feature scaler. None provided."
        # Clean/Scale
        global_feats_reshaped = utilities.clean_features(
            features=global_feats_reshaped, feature_scaler=feature_scaler
        )
        print("global_feats_reshaped", global_feats_reshaped.shape)
        # print(global_feats_reshaped.__class__)
        # print("global_feats_reshaped", global_feats_reshaped)

        if isinstance(data, List):
            cleaned_data = data
            for i in range(len(data)):
                cleaned_data[i].global_feats = torch.as_tensor(global_feats_reshaped[i])

            # print([f"cd.global_feats: {cd.global_feats}" for cd in cleaned_data])

        elif isinstance(data, batch):
            # Flatten and assign back
            # print(
            #     "tensor(global_feats_reshaped).view(-1)",
            #     tensor(global_feats_reshaped).view(-1),
            # )
            cleaned_data.global_feats = torch.as_tensor(global_feats_reshaped).view(-1)

    if add_global_feats_to_nodes:
        if not has_global_feats:
            warnings.warn(
                "There are no global features to add. The data will be returned as is."
            )
        else:
            cleaned_data = add_global_features_to_nodes(cleaned_data)

    return cleaned_data


from rdkit.Chem import (
    BondType,
    Bond,
    BondStereo,
    HybridizationType,
    ChiralType,
    BondType,
    Atom,
)


def decode_atom_rep(
    atom_rep: Union[List, torch.Tensor],
    atom_featurizers_pos_to_values: pd.DataFrame,
    atom_idx: int = None,
):
    if isinstance(atom_rep, torch.Tensor):
        atom_rep = atom_rep.cpu().numpy().tolist()

    def annotate_atom(atom: Chem.Atom, f_type: str, value: int):
        # print(f_type, value)
        # if f_type == 'atomic_num':
        #     atom.SetAtomicNum(value)
        if f_type == "formal_charge":
            atom.SetFormalCharge(value)
        elif f_type == "hybridization":
            atom.SetHybridization(eval(f"HybridizationType.{value.upper()}"))
        elif f_type == "is_aromatic":
            atom.SetIsAromatic(value)
        elif f_type == "chiral_tag" and not value == "unk":
            atom.SetChiralTag(eval(f"ChiralType.{value.upper()}"))
        elif f_type == "n_hydrogens":
            atom.SetNumExplicitHs(value)

    ones_ = [index for index, value in enumerate(atom_rep) if value == 1.0]
    pos_to_values_at_ = atom_featurizers_pos_to_values.iloc[ones_, :]

    # print(pos_to_values_at_.shape)
    atomic_num_ = pos_to_values_at_[pos_to_values_at_["f_type"] == "atomic_num"][
        "value"
    ].values[0]
    atom = Atom(atomic_num_)

    # print(atom.GetAtomicNum(), atom.GetPropsAsDict())
    pos_to_values_at_.apply(
        lambda row: annotate_atom(atom=atom, f_type=row["f_type"], value=row["value"]),
        axis=1,
    )

    return atom


def annotate_bond(
    bond: Bond,
    edge_attr: Union[List, torch.Tensor],
    bond_featurizers_pos_to_values: pd.DataFrame,
    bond_idx: int = None,
):
    def add_bond_props(bond: Chem.Bond, f_type: str, value: int):
        if f_type == "conjugated":
            # print('conjugated', value)
            bond.SetIsConjugated(value)

        elif f_type == "stereo" and not value == "unk":
            # print(eval(f"BondStereo.{value.upper()}"))
            bond.SetStereo(eval(f"BondStereo.{value.upper()}"))
            if value.upper() != "STEREONONE":
                ## explicitly set the atoms involved in the stereochemistry.
                # This can reinforce the stereochemical configuration in cases
                # where RDKit doesn’t automatically assign it based on the atom indices.
                # stereo_atoms = bond.GetStereoAtoms()
                # print("stereo_atoms", list(stereo_atoms))
                # bond.SetStereoAtoms(*stereo_atoms)
                bond.SetStereoAtoms(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                # bond.SetStereoAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                # print("stereo_atoms", list(bond.GetStereoAtoms()))
            # print(f'({bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} {bond.GetBondType()}) - conjugated= {value}") ::  {bond.GetStereo()} was set.')
        elif f_type == "is_aromatic":
            bond.SetIsAromatic(value)
        elif f_type == "chiral_tag":
            bond.SetChiralTag(eval(f"ChiralType.{value.upper()}"))

    if isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.cpu().numpy().tolist()

    if not edge_attr[-1] == 1:
        bond_featurizers_pos_to_values.apply(
            lambda row: add_bond_props(
                bond=bond, f_type=row["f_type"], value=row["value"]
            ),
            axis=1,
        )


def graph_to_molecule(
    graph: Data,
    atom_featurizer: AtomFeaturizer = ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = BOND_FEATURIZER,
) -> AllChem.Mol:
    assert (
        not atom_featurizer.atomic_feats_pos_values is None
    ), "The atom featurizer is lacking the atomic_feats_pos_values attribute."
    assert (
        not bond_featurizer.bond_feats_pos_values is None
    ), "The atom featurizer is lacking the bond_feats_pos_values attribute."

    # print("bond_feats_to_values\n", bond_feats_to_values)

    molecule = Chem.RWMol()
    node_count = graph.x.shape[0]

    edges = [tuple(e) for e in graph.edge_index.t().tolist()]
    edge_count = len(edges)
    # print(f"node_count = {node_count}\nedge_count={edge_count}")

    ## Add atoms and bonds, and annotate them

    atoms, bonds = [], []
    for act in range(node_count):
        molecule.AddAtom(
            decode_atom_rep(
                atom_rep=graph.x[act, :],
                atom_featurizers_pos_to_values=atom_featurizer.atomic_feats_pos_values,
                atom_idx=act,
            )
        )

    added_undirected_edges = []
    # print(edges)
    for bct in range(len(edges)):
        edge_attr_ = graph.edge_attr[bct, :]
        # print("\n")
        # print(edges[bct])
        # print("graph.edge_attr[bct,:]", edge_attr_)
        # print("\tadded_undirected_edges", added_undirected_edges)
        if edge_attr_[-1] == 0.0:
            source_at = edges[bct][0]
            end_at = edges[bct][1]
            # print(f"\t{(source_at, end_at)} already visited? {(source_at, end_at) in added_undirected_edges}")
            bond_ones_ = [
                index for index, value in enumerate(edge_attr_) if value == 1.0
            ]
            bond_pos_to_values_at_ = bond_featurizer.bond_feats_pos_values.iloc[
                bond_ones_, :
            ]
            # print(f"\nbond_pos_to_values_at_  {source_at}-{end_at}\n: ", bond_pos_to_values_at_)
            bond_type = bond_pos_to_values_at_[
                bond_pos_to_values_at_["f_type"] == "bond_type"
            ]["value"].values[0]
            # print("\tMol Get Bonds", molecule.GetNumBonds())

            if not (
                (source_at, end_at) in added_undirected_edges
            ):  # or ((end_at, source_at) in added_undirected_edges):
                added_undirected_edges.extend(
                    [(source_at, end_at), (end_at, source_at)]
                )
                molecule.AddBond(
                    beginAtomIdx=source_at,
                    endAtomIdx=end_at,
                    order=eval(f"BondType.{bond_type.upper()}"),
                )
                # print(f"\t{source_at}-{end_at} and {end_at}-{source_at} were added.")
                # print("\t", [f'{bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} {bond.GetBondType()}' for bond in molecule.GetBonds()])
                annotate_bond(
                    bond=molecule.GetBondBetweenAtoms(source_at, end_at),
                    edge_attr=graph.edge_attr[bct, :],
                    bond_featurizers_pos_to_values=bond_pos_to_values_at_,
                )
    Chem.SanitizeMol(molecule)
    Chem.AssignStereochemistry(molecule)

    return molecule
