import os
import sys
import inspect
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Union, Any, Tuple
from itertools import chain
import random

from sklearn.manifold import TSNE

from rdkit import Chem, DataStructs
from rdkit.Chem import rdmolops
from rdkit.Chem.AllChem import Mol
from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprint,
    GetAtomPairFingerprint,
    GetTopologicalTorsionFingerprint,
)
from rdkit.Chem import (
    PandasTools,
    AllChem,
    MolFromSmiles,
    Draw,
    MolToInchiKey,
    MolToSmiles,
)
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from torch import manual_seed, cuda, backends, Generator

import networkx as nx

def check_if_param_used(cls, param_name):
    signature = inspect.signature(cls.__init__)
    return param_name in signature.parameters

def set_seeds(seed: int = None):
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # controls the hash seed for hash-based operations so they are reproducible, if seed is not None

    random.seed(seed)
    np.random.seed(seed)

    # For general torch operations
    manual_seed(seed)

    # For operations that happen on the CPU
    if cuda.is_available():
        cuda.manual_seed(
            seed
        )  # Sets the seed for generating random numbers for the current GPU.
        cuda.manual_seed_all(
            seed
        )  # Sets the seed for generating random numbers on all GPUs.
        Generator().manual_seed(seed)
        if not seed is None:
            backends.cudnn.deterministic = (
                True  # causes cuDNN to only use deterministic convolution algorithms
            )
        backends.cudnn.benchmark = False  # causes cuDNN to benchmark multiple convolution algorithms and select the fastest

    # torch.use_deterministic_algorithms(True)


def randomize_smiles(smiles, isomericSmiles=False):
    "Take a SMILES string, and return a valid, randomized, abd equivalent SMILES string"
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    random = Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles
    )
    return random


def add_numbers_to_mol_atoms(mol):
    for atom in mol.GetAtoms():
        # For each atom, set the property "atomNote" to a index+1 of the atom
        atom.SetProp("atomNote", str(atom.GetIdx() + 1))


def molecule_from_smiles(smiles: str, add_explicit_h: bool = True):
    ### Modified version of the code from
    ### https://keras.io/examples/graph/mpnn-molecular-graphs/
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    return sanitize_molecule(molecule, add_explicit_h)


def sanitize_molecule(molecule: Mol, add_explicit_h: bool = True):
    if not molecule is None:
        # print(Chem.MolToSmiles(molecule))
        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(
                molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag
            )
        if add_explicit_h:
            molecule = Chem.AddHs(molecule)
            # print(Chem.MolToSmiles(molecule))
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def get_largest_fragment_from_smiles(smiles: str, return_as_smiles: bool = False):
    molecule = MolFromSmiles(smiles)
    return get_largest_fragment_from_mol(molecule, return_as_smiles)


def get_largest_fragment_from_mol(molecule: Mol, return_as_smiles: bool = False):
    if molecule is not None:
        try:
            mol_frags = rdmolops.GetMolFrags(molecule, asMols=True)
            largest_mol = max(
                mol_frags, default=molecule, key=lambda m: m.GetNumAtoms()
            )
            if return_as_smiles:
                largest_mol = MolToSmiles(largest_mol)
            return largest_mol
        except Exception as exp:
            print(f"Could not get largest fragment: {exp}")
    else:
        return None


def augment_data(compounds_df, smiles_column, target_column, n_randomizations=1):
    temp_dfs = []
    for index, row in compounds_df.iterrows():
        original_smiles = row[smiles_column]
        smiles = [original_smiles]
        for i in range(n_randomizations):
            smiles.append(randomize_smiles(original_smiles))
        #         print("SMILES = {}".format(smiles))
        df = pd.DataFrame(list(set(smiles)), columns=[smiles_column])
        df[target_column] = row[target_column]
        #         print(df)
        temp_dfs.append(df)
    final_df = pd.concat(temp_dfs, axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def clean_molecule(
    molecule: Mol, keep_largest_component_only: bool = True, sanitize: bool = True
):
    pass


def min_max_train_test_split_df(
    dataframe,
    molecule_column,
    inchikey_column,
    test_ratio=0.2,
    fp_type="morgan",
    random_state=1,
    return_indices=False,
):
    """ """

    # Store the InChIKeys. These will be used to split the dataframe to ensure
    # no molecule is both in the train and test sets.
    if inchikey_column is None:
        print("Computing and storing the InChiKeys...")
        inchikey_column = "InChIKey"
        dataframe[inchikey_column] = dataframe[molecule_column].apply(
            lambda x: MolToInchiKey(x)
        )

    dataframe.apply(
        lambda x: x[molecule_column].SetProp("InChIKey", x[inchikey_column]), axis=1
    )

    # Select unique molecules (by InChiKey)
    dataframe_single_ikeys = dataframe.drop_duplicates(
        subset=[inchikey_column], keep="first"
    )
    list_of_rdkit_molecules = dataframe_single_ikeys[molecule_column].values.tolist()

    # Split datasets
    print("Splitting the dataset...")
    train_test_splits = min_max_train_test_split(
        list_of_rdkit_molecules,
        test_ratio=test_ratio,
        fp_type=fp_type,
        random_state=random_state,
        return_indices=False,
    )

    train_inchikeys = list(
        set([mol.GetProp(inchikey_column) for mol in train_test_splits[0]])
    )
    test_inchikeys = list(
        set([mol2.GetProp(inchikey_column) for mol2 in train_test_splits[1]])
    )

    print(
        "Train/Test InChiKey Intersection = {}".format(
            [i for i in train_inchikeys if i in test_inchikeys]
        )
    )
    print(
        "Unique InChIKeys:: Train: {} - Test: {}".format(
            len(train_inchikeys), len(test_inchikeys)
        )
    )

    dataframe_train = dataframe[dataframe[inchikey_column].isin(train_inchikeys)]
    dataframe_test = dataframe[dataframe[inchikey_column].isin(test_inchikeys)]
    print("Train: {} - Test: {}".format(dataframe_train.shape, dataframe_test.shape))
    print(dataframe_train.columns)
    return dataframe_train, dataframe_test


def min_max_train_test_split(
    list_of_rdkit_molecules,
    test_ratio,
    fp_type="morgan",
    random_state=1,
    return_indices=False,
):
    """
    fp_types = { "morgan": "GetMorganFingerprint", "atom_pair": "GetAtomPairFingerprint", "top_torso": "GetTopologicalTorsionFingerprint"}
    """

    picker = MaxMinPicker()
    fps = None

    if fp_type == "morgan":
        fps = [GetMorganFingerprint(x, 3) for x in list_of_rdkit_molecules]
    elif fp_type == "atom_pair":
        fps = [GetAtomPairFingerprint(x) for x in list_of_rdkit_molecules]
    elif fp_type == "top_torso":
        fps = [GetTopologicalTorsionFingerprint(x) for x in list_of_rdkit_molecules]

    nfps = len(fps)
    n_training_compounds = round(nfps * (1 - test_ratio))

    ## Calculate the Dice dissimilarity between compounds
    def distij(i, j, fps=fps):
        return 1 - DataStructs.DiceSimilarity(fps[i], fps[j])

    train_indices = picker.LazyPick(
        distij, nfps, n_training_compounds, seed=random_state
    )
    test_indices = [i for i in range(n_training_compounds) if not i in train_indices]

    print("Indices (test): {}".format([x for x in train_indices if x in test_indices]))

    if return_indices:
        return train_indices, test_indices
    else:
        return [list_of_rdkit_molecules[i] for i in train_indices], [
            list_of_rdkit_molecules[j] for j in test_indices
        ]


def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def flatten_list(mylist: List[Any]):
    return list(chain(*mylist))


def flatten(tensor):
    # cpu(): Returns a copy of this object in CPU memory. If this object is already in CPU
    # memory and on the correct device, then no copy is performed and the original object is returned.
    # detach(): Returns a new Tensor, detached from the current graph. The result will never require gradient.
    # numpy(): Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray
    # share the same underlying storage. Changes to self tensor will be reflected in the ndarray and vice versa.
    # flatten(): Flattens input by reshaping it into a one-dimensional tensor.
    return tensor.cpu().detach().numpy().flatten()


def avg_and_drop_duplicates(dataframe, target, inchikey_column):
    groups = []

    for name, group in dataframe.groupby(inchikey_column):
        mean_target_value = group[target].mean()
        #         print("{} - {} - {}".format(group.shape, group[target].values, mean_target_value))
        unique_row = group.drop_duplicates(subset=[inchikey_column], keep="first")
        unique_row[target] = mean_target_value
        #         print("\t{} - {} - {}".format(unique_row.shape, unique_row[target].values, mean_target_value))
        groups.append(unique_row)
    #     print(groups)
    return pd.concat(groups, axis=0)


def remove_conflicting_target_values(dataframe, target, inchikey_column):
    groups = []

    for name, group in dataframe.groupby(inchikey_column):
        n_conflicts = 0
        unique_target_values = group[target].unique().tolist()
        if len(unique_target_values) > 1:
            n_conflicts += 1
            print(
                "InchIKey {} has {} conflicting {} values. All associated samples will be removed.".format(
                    name, len(unique_target_values), target
                )
            )
        else:
            #         print("{} - {} - {}".format(group.shape, group[target].values, mean_target_value))
            unique_row = group.drop_duplicates(subset=[inchikey_column], keep="first")
            groups.append(unique_row)
    #     print(groups)
    if n_conflicts > 0:
        print("Number of unique compounds with conflicts: {}".format(n_conflicts))
    return pd.concat(groups, axis=0)


def min_max_train_validate_test_split_df(
    dataframe,
    molecule_column,
    inchikey_column=None,
    fp_column=None,
    train_valid_ratios=[0.7, 0.15],
    fp_type="morgan",
    random_state=1,
    return_indices=False,
):
    """ """

    # Store the InChIKeys. These will be used to split the dataframe to ensure no molecule is both in the train and test sets.
    if inchikey_column is None:
        print("Computing and storing the InChiKeys...")
        inchikey_column = "InChIKey"
        dataframe[inchikey_column] = dataframe[molecule_column].apply(
            lambda x: MolToInchiKey(x)
        )

    dataframe.apply(
        lambda x: x[molecule_column].SetProp("InChIKey", x[inchikey_column]), axis=1
    )

    # Select unique molecules (by InChiKey)
    dataframe_single_ikeys = dataframe.drop_duplicates(
        subset=[inchikey_column], keep="first"
    )
    list_of_rdkit_representations = None
    if fp_column is not None:
        list_of_rdkit_representations = dataframe_single_ikeys[
            fp_column
        ].values.tolist()
    else:
        list_of_rdkit_representations = dataframe_single_ikeys[
            molecule_column
        ].values.tolist()

    # Split datasets
    print("Splitting the dataset...")
    train_validate_test_splits = min_max_train_validate_test_split(
        list_of_rdkit_representations,
        train_valid_ratios=train_valid_ratios,
        fp_type=fp_type,
        random_state=random_state,
        return_indices=True,
    )

    #     print("Train: {} - Validate: {} - Test: {}".format(train_validate_test_splits[0], train_validate_test_splits[1], train_validate_test_splits[2]))
    train_inchikeys = list(
        set(
            dataframe.iloc[train_validate_test_splits[0]][
                inchikey_column
            ].values.tolist()
        )
    )
    validate_inchikeys = list(
        set(
            dataframe.iloc[train_validate_test_splits[1]][
                inchikey_column
            ].values.tolist()
        )
    )
    test_inchikeys = list(
        set(
            dataframe.iloc[train_validate_test_splits[2]][
                inchikey_column
            ].values.tolist()
        )
    )

    dataframe_train = dataframe[dataframe[inchikey_column].isin(train_inchikeys)]
    dataframe_validate = dataframe[dataframe[inchikey_column].isin(validate_inchikeys)]
    dataframe_test = dataframe[dataframe[inchikey_column].isin(test_inchikeys)]
    print(
        "Train: {} - Validate: {} - Test: {}".format(
            dataframe_train.shape, dataframe_validate.shape, dataframe_test.shape
        )
    )
    print(dataframe_train.columns)
    return dataframe_train, dataframe_validate, dataframe_test


def min_max_train_validate_test_split(
    list_of_rdkit_representations,
    train_valid_ratios=[0.7, 0.15],
    fp_type="morgan",
    random_state=1,
    return_indices=False,
):
    """
    fp_types = { "morgan": "GetMorganFingerprint", "atom_pair": "GetAtomPairFingerprint", "top_torso": "GetTopologicalTorsionFingerprint"}
    """
    try:
        input_mode = list_of_rdkit_representations[0].__class__.__name__

        picker = MaxMinPicker()
        fps = None
        list_of_rdkit_representations = [
            x for x in list_of_rdkit_representations if not x is None
        ]
        orginal_indices = range(len(list_of_rdkit_representations))
        fps = None

        if input_mode == "Mol":
            if fp_type == "morgan":
                fps = [
                    GetMorganFingerprint(x, 3) for x in list_of_rdkit_representations
                ]
            elif fp_type == "atom_pair":
                fps = [GetAtomPairFingerprint(x) for x in list_of_rdkit_representations]
            elif fp_type == "top_torso":
                fps = [
                    GetTopologicalTorsionFingerprint(x)
                    for x in list_of_rdkit_representations
                ]
        #             elif fp_type is None and fp_column is not None:
        #                 fps = [mol.GetProp(fp_column).strip('][').split(', ') for mol in list_of_rdkit_representations]
        #                 for i in fps:
        #                     for j in range(len(i)):
        #                         i[j] = int(i[j])
        elif input_mode in ["UIntSparseIntVect", "SparseIntVect", "ExplicitBitVect"]:
            fps = list_of_rdkit_representations

        if fps is not None:
            nfps = len(fps)
            n_training_compounds = round(nfps * (train_valid_ratios[0]))
            n_valid_compounds = round(nfps * (train_valid_ratios[1]))
            n_test_compounds = nfps - n_training_compounds - n_valid_compounds
            print(
                "{} - {} - {}".format(
                    n_training_compounds, n_valid_compounds, n_test_compounds
                )
            )

            ## Calculate the Dice dissimilarity between compounds
            def distij(i, j, fps=fps):
                return 1 - DataStructs.DiceSimilarity(fps[i], fps[j])

            ## Retrieving training indices
            training_indices = list(
                picker.LazyPick(distij, nfps, n_training_compounds, seed=random_state)
            )
            #         print(training_indices)

            ## Retrieving validation indices
            remaining_indices = [
                x for x in orginal_indices if not x in training_indices
            ]
            fps = [fps[j] for j in remaining_indices]
            nfps = len(fps)
            #         print("reamining: {}".format(nfps))
            val = list(
                picker.LazyPick(distij, nfps, n_valid_compounds, seed=random_state)
            )
            #         print(val)
            validation_indices = [remaining_indices[k] for k in val]

            ## Retrieving test indices
            test_indices = [
                l
                for l in orginal_indices
                if not l in training_indices + validation_indices
            ]

            print(
                "Indices (training):{} - {}".format(
                    len(training_indices), training_indices[:2]
                )
            )
            print(
                "Indices (validation):{} - {}".format(
                    len(validation_indices), validation_indices[:1]
                )
            )
            print("Indices (test):{} - {}".format(len(test_indices), test_indices[:1]))

            if return_indices:
                return training_indices, validation_indices, test_indices
            else:
                return (
                    [list_of_rdkit_representations[i] for i in training_indices],
                    [list_of_rdkit_representations[j] for j in validation_indices],
                    [list_of_rdkit_representations[j] for j in test_indices],
                )
        else:
            raise ValueError(
                "Could not perform clustering and selection.\tFingerprint list = None"
            )
    except Exception as e:
        print("Could not perform clustering and selection.")
        print(e)
        return None


#######################################
#             VISUALIZATION           #
#######################################
def plots_train_val_metrics(
    train_losses: List[float],
    val_scores: List[float],
    val_losses: List[float] = None,
    figsize: Tuple = (10, 7),
    image_pathname: str = None,
    val_score_name: str = None,
):
    plt.figure(figsize=figsize)
    plt.plot(train_losses, color="orange", label="train loss")
    if not val_losses is None:
        plt.plot(val_losses, color="red", label="val. loss")
    val_score_label = (
        "val. score" if val_score_name is None else f"val. score ({val_score_name})"
    )

    plt.plot(val_scores, color="green", label=val_score_label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Score")
    plt.legend()
    if not image_pathname is None:
        plt.savefig(image_pathname)
    plt.show()


def plot_embeddings(model, data):
    """
    From https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html#iii.-implementing-a-graph-attention-network
    """

    h = model(data)

    # Train TSNE
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(
        h.detach()
    )

    # Plot TSNE
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
    plt.show()


def visualize_colored_graph(edge_index, edge_mask, data, threshold=0.5):
    # Build graph from edge_index
    G = nx.Graph()
    edge_list = edge_index.t().tolist()  # Convert edge_index to list of edges

    # Remove self-loops (edges where source and target nodes are the same)
    edge_list = [edge for edge in edge_list if edge[0] != edge[1]]

    G.add_edges_from(edge_list)

    # Convert edge_mask to numpy and resize if necessary
    edge_mask = edge_mask.cpu().detach().numpy()  # Convert tensor to numpy
    if len(edge_mask) != len(edge_list):
        edge_mask = edge_mask[: len(edge_list)]

    important_edges = edge_mask > threshold

    # Normalize the edge_mask for coloring
    edge_mask_normalized = edge_mask / edge_mask.max()  # Normalize to [0, 1]

    # Create the list of edge colors using colormap
    cmap = plt.cm.Reds  # Colormap to use
    edge_colors = cmap(edge_mask_normalized)  # Get colors for edges

    # Filter the edges based on importance (threshold)
    edges_to_draw = np.array(edge_list)[important_edges]
    edge_colors_to_draw = edge_colors[important_edges]

    # Get node positions
    pos = nx.spring_layout(G)  # Position the graph layout
    plt.figure(figsize=(8, 6))  # Set figure size

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)

    # Draw edges with corresponding colors
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_to_draw,
        edge_color=edge_colors_to_draw,
        edge_cmap=cmap,
        width=2,
        edge_vmin=0,
        edge_vmax=1,
    )

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # Set the array to use for color mapping

    # Create a new Axes for colorbar
    ax_colorbar = plt.axes(
        [0.95, 0.12, 0.03, 0.75]
    )  # Adjust the position and size of the colorbar
    plt.colorbar(sm, cax=ax_colorbar)  # Create the colorbar

    plt.title("Graph with Edge Importance")
    plt.show()
