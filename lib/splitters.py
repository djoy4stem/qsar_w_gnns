from itertools import chain, accumulate
from math import ceil, floor, isclose
import pandas as pd
from pandas.core import series
import numpy as np
from random import Random

from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from rdkit.Chem import (
    AllChem,
    PandasTools,
    MolToInchiKey,
    MolToSmiles,
    MolFromSmiles,
    SanitizeMol,
    rdFingerprintGenerator,
)

from rdkit.Chem.AllChem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from typing import Any, List, Union

from lib import utilities


def check_ratios(
    train_ratio: float = 0.8, val_ratio: float = None, test_ratio: float = 0.2
):
    val_ratio = float(val_ratio or 0)
    # if val_ratio is None:
    #     assert isclose(0.9999999, train_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06), f"train_ratio + test_ratio must be equals 1, not {train_ratio + test_ratio}."
    # else:
    assert isclose(
        0.9999999, train_ratio + val_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06
    ), f"train_ratio + val_ratio + test_ratio must be equals 1, not {train_ratio + val_ratio + test_ratio}."


def flatten(mylist: List[Any]):
    return [item for level_2_list in mylist for item in level_2_list]


class ScaffoldSplitter(object):
    @staticmethod
    def get_bemis_murcko_scaffolds(
        molecules: List[Mol],
        include_chirality: bool = False,
        return_as_indices: bool = True,
        sort_by_size: bool = True,
    ):
        def bm_scaffold(molecule: Mol, include_chirality: bool = False):
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=molecule, includeChirality=include_chirality
                )

                return scaffold
            except Exception as exp:
                print(
                    f"Could not generate a Bemis-Murck scaffold for the query molecule. \n{exp}"
                )
                # return None

        try:
            scaffold_smiles = [bm_scaffold(mol, include_chirality) for mol in molecules]

            scaffolds = {}

            for s in range(len(scaffold_smiles)):
                scaf = scaffold_smiles[s]
                if scaf in scaffolds:
                    scaffolds[scaf].append(s)
                else:
                    scaffolds[scaf] = [s]

            for skey in scaffolds:
                scaffolds[skey].sort()

            if sort_by_size:
                ## Sort by decreasing number of molecules that have a given scaffold
                scaffolds = dict(
                    sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
                )

            if return_as_indices:
                return scaffolds

            else:
                scaffolds_items = list(scaffolds.items())
                my_scaffolds = {}

                for j in range(len(scaffolds_items)):
                    mols = None
                    if isinstance(mols, series.Series):
                        mols = [molecules.iloc[k] for k in scaffolds_items[j][1]]
                    else:
                        mols = [molecules[k] for k in scaffolds_items[j][1]]

                    my_scaffolds[scaffolds_items[j][0]] = mols

                return my_scaffolds
        except Exception as exp:
            print(
                "Could not generate Bemis-Murcko scaffolds for the list of molecules."
            )
            raise Exception(exp)

    @staticmethod
    def train_val_test_split(
        molecules: List[Mol],
        val_ratio: float = None,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        return_as_indices: bool = False,
        return_as_clusters: bool = False,
        include_chirality: bool = False,
        sort_by_size: bool = True,
        shuffle_idx: bool = False,
        random_state: int = 1,
    ):
        def len_for_list_of_dicts(ldicts: List[dict]):
            # print([len(d[1]) for d in ldicts])
            l = sum([len(d[1]) for d in ldicts])
            return l

        # try:
        if True:
            check_ratios(
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
            )

            train_size = train_ratio * len(molecules)
            val_size = float(val_ratio or 0) * len(molecules)
            test_size = len(molecules) - train_size - val_size

            train_scaffolds, val_scaffolds, test_scaffolds = [], [], []

            bmscaffolds = ScaffoldSplitter.get_bemis_murcko_scaffolds(
                molecules=molecules,
                include_chirality=include_chirality,
                return_as_indices=return_as_indices,
                sort_by_size=sort_by_size,
            )

            # print("bmscaffolds = ", bmscaffolds)
            curr_train_len, curr_val_len, curr_test_len = 0, 0, 0

            bmscaffolds_items = list(bmscaffolds.items())
            # print(bmscaffolds_items[:10])

            if shuffle_idx:
                a = bmscaffolds_items[10:]
                Random(random_state).shuffle(a)
                bmscaffolds_items = bmscaffolds_items[:10] + a

            for bms in bmscaffolds_items:
                # print(train_scaffolds)
                # print(len(bms[1]), bms)
                # print(f"curr_train_len: {curr_train_len}")
                # print(f"len(bms): {len(bms[1])}")
                bms_size = len(bms[1])
                if curr_train_len + bms_size > train_size:
                    if val_size > 0:
                        if curr_val_len + bms_size > val_size:
                            # print(f"adding bms to test")
                            test_scaffolds.append(bms)

                        else:
                            # print(f"adding bms to val: {curr_val_len + bms_size}")
                            val_scaffolds.append(bms)
                            curr_val_len = len_for_list_of_dicts(val_scaffolds)
                else:
                    # print(f"adding bms to train: {curr_train_len + bms_size}")
                    train_scaffolds.append(bms)
                    curr_train_len = len_for_list_of_dicts(train_scaffolds)

            if not return_as_clusters:
                train_scaffolds = utilities.flatten_list(
                    [t[1] for t in train_scaffolds]
                )
                val_scaffolds = utilities.flatten_list([t[1] for t in val_scaffolds])
                test_scaffolds = utilities.flatten_list([t[1] for t in test_scaffolds])

            return train_scaffolds, val_scaffolds, test_scaffolds
        # except Exception as exp:
        #     print(f"Could not split dataset. \n{exp}")

    @staticmethod
    def kfold_split(
        molecules: List[Mol],
        n_folds: int = 5,
        return_as_indices: bool = False,
        include_chirality: bool = False,
        random_state: int = 1,
        sort_by_size: bool = True,
    ):
        try:
            # if True:
            fold_size = ceil(len(molecules) / n_folds)
            folds = []
            start_idx = 0
            for i in range(n_folds - 1):
                print(
                    f"Fold {i}  :: start: {start_idx} - stop: {start_idx+fold_size-1}"
                )
                if isinstance(molecules, series.Series):
                    folds.append(molecules.iloc[start_idx : start_idx + fold_size - 1])
                else:
                    folds.append(molecules[start_idx : start_idx + fold_size - 1])

                start_idx += fold_size - 1

            print(f"start_idx: {start_idx}")
            folds.append(molecules[start_idx:])

            return folds
        except Exception as exp:
            print(f"Could not perform k-fold split on dataset. \n{exp}")


class ClusterSplitter(object):
    _allowable_fptypes = ["morgan", "atom_pair", "topoligical"]

    @staticmethod
    def generate_fingerprints(
        molecules: List[AllChem.Mol],
        fingerprint_type: str = "morgan",
        fp_size: int = 1024,
        radius: int = 2,
        count_bits: bool = True,
        include_chirality: bool = False,
    ):
        fingerprints = []
        fprinter = None
        if fingerprint_type == "morgan":
            fprinter = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=fp_size, countSimulation=count_bits
            )
        elif fingerprint_type == "atom_pair":
            fprinter = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=fp_size,
                countSimulation=count_bits,
                maxPath=30,
                includeChirality=include_chirality,
            )
        elif fingerprint_type == "topoligical":
            fprinter = rdFingerprintGenerator.GetTopologicalTorsionalGenerator(
                fpSize=fp_size,
                countSimulation=count_bits,
                includeChirality=include_chirality,
            )
        else:
            raise ValueError(
                f"No implementation for fingerprint type {fingerprint_type}. Allowable types include: {'; '.join(_allowable_fptypes)}"
            )

        if count_bits:
            for mol in molecules:
                try:
                    fingerprints.append(fprinter.GetCountFingerprint(mol))
                except:
                    fingerprints.append(None)
        else:
            for mol in molecules:
                try:
                    fingerprints.append(fprinter.GetFingerprint(mol))
                except:
                    fingerprints.append(None)

        return fingerprints

    @staticmethod
    def tanimoto_distance_matrix(fp_list):
        """
        Calculate distance matrix for fingerprint list
        Taken from https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html

        """
        dissimilarity_matrix = []
        # Notice how we are deliberately skipping the first and last items in the list
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    @staticmethod
    def cluster_fingerprints(fingerprints, sim_cutoff=0.2):
        """Cluster fingerprints
        Parameters:
            fingerprints
            dist_threshold: threshold for the clustering. Molecules with a dissimilarity below (or similarity above) this threshold are grouped into the same cluster.
        """
        # Calculate Tanimoto distance matrix
        distance_matrix = ClusterSplitter.tanimoto_distance_matrix(fingerprints)
        # Now cluster the data with the implemented Butina algorithm:
        clusters = Butina.ClusterData(
            data=distance_matrix,
            nPts=len(fingerprints),
            distThresh=1 - sim_cutoff,
            isDistData=True,
        )
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters

    @staticmethod
    def cluster_molecules(
        molecules: Union[List[AllChem.Mol], pd.Series],
        fingerprint_type: str = "morgan",
        fp_size: int = 1024,
        radius: int = 2,
        count_bits: bool = True,
        include_chirality: bool = False,
        sim_cutoff=0.7,
        return_idx: bool = False,
    ):
        fingerprints = ClusterSplitter.generate_fingerprints(
            molecules=molecules,
            fingerprint_type=fingerprint_type,
            fp_size=fp_size,
            radius=radius,
            count_bits=count_bits,
            include_chirality=include_chirality,
        )

        clusters = ClusterSplitter.cluster_fingerprints(
            fingerprints=fingerprints, sim_cutoff=sim_cutoff
        )

        if return_idx:
            return clusters
        else:
            clusters_of_mols = []
            for cluster in clusters:
                cluster_m = None
                if isinstance(molecules, series.Series):
                    cluster_m = [molecules.iloc[i] for i in cluster]
                else:
                    cluster_m = [molecules[i] for i in cluster]
                clusters_of_mols.append(cluster_m)

            return clusters_of_mols

    @staticmethod
    def train_val_test_split(
        molecules: List[Mol],
        val_ratio: float = None,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        return_as_indices: bool = False,
        return_as_clusters: bool = False,
        include_chirality: bool = False,
        fingerprint_type: str = "morgan",
        fp_size: int = 1024,
        radius: int = 2,
        count_bits: bool = True,
        sim_cutoff=0.2,
        sort_by_size: bool = True,
        shuffle_idx: bool = False,
        random_state: int = 1,
    ):
        def len_for_list_of_dicts(ldicts: List[dict]):
            # print([len(d[1]) for d in ldicts])
            l = sum([len(d) for d in ldicts])
            return l

        # try:
        if True:
            check_ratios(
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
            )

            train_size = train_ratio * len(molecules)
            val_size = float(val_ratio or 0) * len(molecules)
            test_size = len(molecules) - train_size - val_size
            print(train_size, val_size, test_size)

            train_clusters, val_clusters, test_clusters = [], [], []

            clusters = ClusterSplitter.cluster_molecules(
                molecules=molecules,
                fingerprint_type=fingerprint_type,
                fp_size=fp_size,
                radius=radius,
                count_bits=count_bits,
                include_chirality=include_chirality,
                sim_cutoff=sim_cutoff,
                return_idx=return_as_indices,
            )

            # print("clusters = ", clusters)
            # print([len(cl) for cl in clusters])
            curr_train_len, curr_val_len, curr_test_len = 0, 0, 0

            # print(clusters[:10])

            if shuffle_idx:
                a = clusters[10:]
                Random(random_state).shuffle(a)
                clusters = clusters[:10] + a

            bms_counter = 0
            # not_added = []
            for bms in clusters:
                bms_counter += 1
                # print(train_clusters)
                # print(len(bms[1]), bms)
                # print(f"bms len: {len(bms)}  -- curr_train_len: {curr_train_len}")
                # print(f"len(bms): {len(bms[1])}")
                bms_size = len(bms)
                if curr_train_len + bms_size > train_size and bms_counter > 1:
                    # print("OK")
                    # if  bms_counter == 1:
                    #     train_clusters.append(bms)
                    #     curr_train_len = len_for_list_of_dicts(train_clusters)
                    if val_size > 0:
                        if curr_val_len + bms_size > val_size:
                            # print(f"adding bms to test")
                            test_clusters.append(bms)

                        else:
                            # print(f"adding bms to val: {curr_val_len + bms_size}")
                            val_clusters.append(bms)
                            curr_val_len = len_for_list_of_dicts(val_clusters)
                    else:
                        # print(f"adding bms to test: {curr_test_len + bms_size}")
                        test_clusters.append(bms)
                        curr_test_len = len_for_list_of_dicts(test_clusters)
                else:
                    # print(f"adding bms to train: {curr_train_len + bms_size}")
                    train_clusters.append(bms)
                    curr_train_len = len_for_list_of_dicts(train_clusters)

            if not return_as_clusters:
                train_clusters = utilities.flatten_list(train_clusters)
                val_clusters = utilities.flatten_list(val_clusters)
                test_clusters = utilities.flatten_list(test_clusters)

            return train_clusters, val_clusters, test_clusters
        # except Exception as exp:
        #     print(f"Could not split dataset. \n{exp}")
