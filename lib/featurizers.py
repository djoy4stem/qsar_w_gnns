import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AddHs,
    Draw,
    AllChem,
    Crippen,
    QED,
    Descriptors,
    GraphDescriptors,
    Fragments,
    MolToSmiles,
    GetPeriodicTable,
)
from rdkit.Chem import rdMolDescriptors as rdmdesc
from typing import List, Any
from lib import utilities
import re
import mordred
from mordred import descriptors, Calculator
from time import time

RDLogger.DisableLog("rdApp.*")


RDKIT_FRAGS = [
    [frag_name, MolToSmiles(eval(f"Fragments.{frag_name}").__defaults__[1])]
    for frag_name in dir(Fragments)
    if not re.match("^fr", frag_name) is None
]
DF_FUNC_GRPS = pd.DataFrame(RDKIT_FRAGS, columns=["name", "SMARTS"])


## Some examples were taken from https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
DF_FUNC_GRPS_MINI = pd.DataFrame(
    [
        ["aldehyde", "[$([CX3H][#6]),$([CX3H2])]=[OX1]"],
        ["carbonyl", "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]"],
        ["ketone", "[#6][CX3](=[OX1])[#6]"],
        ["carboxyl", "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]"],
        ["aryl_carboxy", "[#6;a](=[OX1])[$([OX2H]),$([OX1-])]"],
        ["n_nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][#7]"],
        [
            "carboxylic_ester",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
        ],
        [
            "ether",
            "[OX2]([#6;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])])[#6;!$(C([OX2])[O,S,#7,#15])]",
        ],
        [
            "thioether",
            "[SX2]([#6;!$(C([SX2])[O,S,#7,#15,F,Cl,Br,I])])[#6;!$(C([SX2])[O,S,#7,#15])]",
        ],
        ["lactone", "[#6][#6X3R](=[OX1])[#8X2][#6;!$(C=[O,N,S])]"],
        [
            "lactam",
            "	[#6R][#6X3R](=[OX1])[#7X3;$([H1][#6;!$(C=[O,N,S])]),$([H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        ["alcohol", "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"],
        ["phenol", "[OX2H][c]"],
        ["amine", "[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]"],
        ["arylhalide", "[Cl,F,I,Br][c]"],
        ["alkylhalide", "[Cl,F,I,Br][CX4]"],
        ["urea", "[#7X3;!$([#7][!#6])][#6X3](=[OX1])[#7X3;!$([#7][!#6])]"],
        ["thiourea", "[#7X3;!$([#7][!#6])][#6X3](=[SX1])[#7X3;!$([#7][!#6])]"],
        [
            "sulfoniamide",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        ["carbamic_acid", "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]"],
        ["azo_nitrogen", "[NX2]=N"],
        ["azoxy_nitrogen", "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]"],
        ["diazo_nitrogen", "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]"],
        ["azole", "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]"],
        ["nitro_group", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]"],
        ["carbo_thiocarboxylate", "[S-][CX3](=S)[#6]"],
        ["sulfone", "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]"],
        [
            "sulfonamide",
            "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        ],
        ["sp3_nitrogen", "[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]"],
        ["unbranched_alkane", "[R0;D2][R0;D2][R0;D2][R0;D2]"],
        ["unbranched_chain", "[R0;D2]~[R0;D2]~[R0;D2]~[R0;D2]"],
        ["para_subs_ring", "*-!:aaaa-!:*"],
        ["meta_subs_ring", "*-!:aaa-!:*"],
        ["ortho_subs_ring", "*-!:aa-!:*"],
        ["azide_ion", "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]"],
        ["N#S", "N#S"],
        ["N=S", "N=S"],
        ["S=S", "S=S"],
        ["C#S", "C#S"],
        ["michael_acceptor", "[CX3]=[CX3][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-])]"]
    ],
    columns=["name", "SMARTS"],
)


### Modified version of the code from
### https://keras.io/examples/graph/mpnn-molecular-graphs/
class Featurizer:
    def __init__(self, allowable_sets_one_hot, continuous_props=None):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets_one_hot.items():
            # print(k, s)
            # print(f"sorted(list(s)) = {sorted(list(s))}")
            # if sorted(list(s)) == [False, True]:
            #     s = sorted(list(s))
            #     self.dim += len(s)
            #     # print("s=", s)
            #     self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim))) ## The + 1 marks a bit that will be populated with -1 is the value is not allowed
            # else:
            # print(f"{k}: list(s)", list(s))
            s = sorted(list(s)) + ["unk"]
            # print("s=", s)
            self.features_mapping[k] = dict(
                zip(s, range(self.dim, len(s) + self.dim + 1))
            )  ## The + 1 marks a bit that will be populated with -1 is the value is not allowed
            # print("==>", self.features_mapping[k])
            self.dim += len(s)
            # print("==>", self.dim)
        # print("continuous_props", continuous_props)
        if not continuous_props is None:
            for ix, p in enumerate(continuous_props):
                self.features_mapping[p] = self.dim
                self.dim += 1
                # print(self.features_mapping[p])

        # print("==>", self.features_mapping)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        # print(output)
        for name_feature, feature_mapping in self.features_mapping.items():
            # print(name_feature, feature_mapping)
            feature = getattr(self, name_feature)(inputs)  ## e.g.: atomic_num(inputs)
            # print("feature: ", feature)
            if isinstance(feature_mapping, dict):
                if feature not in feature_mapping:
                    output[feature_mapping["unk"]] = 1.0
                else:
                    output[feature_mapping[feature]] = 1.0
            elif isinstance(feature_mapping, int):
                # print(feature_mapping)
                output[feature_mapping] = feature
            else:
                raise TypeError(
                    f"Feature {feature} must have either a dictionary or integer mapping. However, we have a {feature_mapping.__class__} object."
                )
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets_one_hot, continuous_props=None):
        super().__init__(allowable_sets_one_hot, continuous_props)

    def atomic_num(self, atom: Chem.rdchem.Atom):
        return atom.GetAtomicNum()

    def n_valence(self, atom: Chem.rdchem.Atom):
        return atom.GetTotalValence()

    def formal_charge(self, atom: Chem.rdchem.Atom):
        return int(atom.GetFormalCharge())

    def n_hydrogens(self, atom: Chem.rdchem.Atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom: Chem.rdchem.Atom):
        return atom.GetHybridization().name.lower()

    def chiral_tag(self, atom: Chem.rdchem.Atom):
        return int(atom.GetChiralTag())

    def is_aromatic(self, atom: Chem.rdchem.Atom):
        return atom.GetIsAromatic()

    def is_in_ring(self, atom):
        return atom.IsInRing()

    def is_in_ring_size_4(self, atom):
        return atom.IsInRingSize(4)

    def is_in_ring_size_5(self, atom):
        return atom.IsInRingSize(5)

    def is_in_ring_size_6(self, atom):
        return atom.IsInRingSize(6)

    def atomic_mass(self, atom):
        return atom.GetMass()

    def atomic_vdw_radius(self, atom):
        return GetPeriodicTable().GetRvdw(atom.GetAtomicNum())

    def atomic_covalent_radius(self, atom):
        return GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())

    # def electronegativity(self, atom):
    #     return float(GetPeriodicTable().GetRdkitAtom(atom.GetAtomicNum()).GetProp('_PaulingElectronegativity'))


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets_one_hot):
        super().__init__(allowable_sets_one_hot)
        self.dim += 1

    def encode(self, bond: Chem.rdchem.Bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond: Chem.rdchem.Bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond: Chem.rdchem.Bond):
        return bond.GetIsConjugated()

    def stereo(self, bond: Chem.rdchem.Bond):
        return bond.GetStereo().name.lower()

    def is_in_ring(self, bond: Chem.rdchem.Bond):
        return bond.IsInRing()

    def is_in_ring_size_5(self, bond: Chem.rdchem.Bond):
        return bond.IsInRingSize(5)

    def is_in_ring_size_6(self, bond: Chem.rdchem.Bond):
        return bond.IsInRingSize(6)


class MoleculeFeaturizer(object):
    def __init__(
        self,
        features: List[str] = None,
        df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
        label_col: str = "name",
    ):
        self.dim = 0
        self.allowable_calc_features = [
            "CalcExactMolWt",
            "CalcTPSA",
            "CalcNumAromaticRings",
            "CalcNumHBA",
            "CalcNumHBD",
            "CalcNumRotatableBonds"
            # 'CalcLabuteASA',  ## Taken care of with mordred's MoeType
            # , 'CalcAsphericity'
            # , 'CalcPBF'            ## Plane of Best Fit to quantify and characterize the 3D character of molecules.
            # , 'GetUSRScore' ## Ultrafast Shape Recognition (USR) is an alignment-free LBVS technique
            # , 'SlogP_VSA_' ## Taken care of with mordred's MoeType
            # , 'SMR_VSA_' ## Taken care of with mordred's MoeType
            ,
            "CalcChi0n",
            "CalcChi0v",
            "CalcChi1n",
            "CalcChi1v",
            "CalcChi2n",
            "CalcChi2v",
            "CalcChi3n",
            "CalcChi3v",
            "CalcChi4n",
            "CalcChi4v",
            "CalcChi0n",
            "CalcChi0v"
            # , 'CalcChiNn', 'CalcChiNv'
        ]  #'CalcNumAtomStereoCenters',
        self.allowable_crippen_features = ["MolLogP"]
        self.allowable_qed_features = ["qed"]
        self.allowable_descriptors = ["MaxPartialCharge", "MinPartialCharge"]
        self.allowable_graph_descriptors = ["HallKierAlpha"]
        self.default_mordred_descs = [
            "ExtendedTopochemicalAtom",
            "Polarizability",
            "ZagrebIndex",
            "MoeType",
        ]
        self.df_func_gps = df_func_gps
        self.fgcp_label_col = label_col
        # print('self.df_func_gps', self.df_func_gps)

        if not features is None:
            self.rdkit_features = [
                f
                for f in features
                if f
                in self.allowable_calc_features
                + self.allowable_crippen_features
                + self.allowable_qed_features
                + self.allowable_descriptors
                + self.allowable_graph_descriptors
                # + self.mordred_descs
            ]

            # print('features', features)
            self.mordred_descs = [
                f
                for f in features
                if f in dir(mordred.descriptors)
                if re.match("^_", f) is None and f != "all"
            ]
        else:
            self.rdkit_features = (
                self.allowable_calc_features
                + self.allowable_crippen_features
                + self.allowable_qed_features
                + self.allowable_descriptors
                + self.allowable_graph_descriptors
            )

            self.mordred_descs = self.default_mordred_descs

        # print('self.rdkit_features ', self.rdkit_features )
        # print('self.mordred_descs ', self.mordred_descs )

    def compute_rdkit_properties(
        self,
        molecule,
        features: List[str] = None,
        label_col: str = "name",
        as_dict: bool = True,
    ):
        properties = {}
        mordred_features, mordred_descs = [], None
        len_func_groups = 0

        failed_com_preps = []

        if features is None:
            features = self.rdkit_features

        if not molecule is None:
            if not (features is None or len(features) == 0):
                for prop in features:
                    try:
                        if prop in self.allowable_calc_features:
                            properties[prop] = eval(f"rdmdesc.{prop}")(molecule)
                        elif prop in self.allowable_crippen_features:
                            properties[prop] = eval(f"Crippen.{prop}")(molecule)
                        elif prop in self.allowable_qed_features:
                            properties[prop] = eval(f"QED.{prop}")(molecule)
                        elif prop in self.allowable_descriptors:
                            properties[prop] = eval(f"Descriptors.{prop}")(molecule)
                        elif prop in self.allowable_graph_descriptors:
                            properties[prop] = eval(f"GraphDescriptors.{prop}")(
                                molecule
                            )
                    except:
                        # print(f"Could not compute molecular property '{prop}'.")
                        failed_com_preps.append(prop)
                        properties[prop] = None

                if len(failed_com_preps) > 0:
                    print(
                        f"Could not compute molecular properties: {'; '.join(failed_com_preps)}"
                    )

            fcgps = None

            if not self.df_func_gps is None:
                fcgps = get_func_groups_pos_from_mol(
                    molecule,
                    df_func_gps=self.df_func_gps,
                    as_dict=True,
                    label_col=self.fgcp_label_col,
                )

                if fcgps is None and (not features is None):
                    fcgps = dict(
                        zip(
                            self.df_func_gps[self.fgcp_label_col].tolist(),
                            [None] * self.df_func_gps.shape[0],
                        )
                    )

                if not fcgps is None:
                    properties = dict(properties, **fcgps)
                    len_func_groups = len(fcgps)

            # print(f"properties = {properties}")
            if bool(properties):
                if not as_dict:
                    prop_values = []

                    for i in list(properties.values()):
                        if (
                            isinstance(i, float)
                            or isinstance(i, int)
                            or isinstance(i, bool)
                        ):
                            prop_values.append(i)
                        elif (
                            isinstance(i, list)
                            or isinstance(i, tuple)
                            or isinstance(i, set)
                        ):
                            prop_values += list(i)

                    return prop_values
                else:
                    # print("properties 2", properties)
                    return properties
                # print('prop_values', prop_values)
            else:
                return None

        else:
            for f in features:
                properties[f] = None

            if not self.df_func_gps is None:
                properties = dict(
                    properties,
                    **dict(
                        zip(
                            self.df_func_gps[self.fgcp_label_col].tolist(),
                            [None] * self.df_func_gps.shape[0],
                        )
                    ),
                )

            if not as_dict:
                prop_values = []

                for i in list(properties.values()):
                    if (
                        isinstance(i, float)
                        or isinstance(i, int)
                        or isinstance(i, bool)
                    ):
                        prop_values.append(i)
                    elif (
                        isinstance(i, list)
                        or isinstance(i, tuple)
                        or isinstance(i, set)
                    ):
                        prop_values += list(i)

                return prop_values
            else:
                return properties

    def compute_mordred_props(self, molecules, mordred_props: list = None):
        try:
            clean_props = None
            # print('mordred_props', mordred_props)
            if not mordred_props is None:
                clean_props = [eval(f"descriptors.{prop}") for prop in mordred_props]
            elif not len(self.mordred_descs) == 0:
                clean_props = [
                    eval(f"descriptors.{prop}") for prop in self.default_mordred_descs
                ]

            # print('clean_props', clean_props)
            if not (clean_props is None or len(clean_props) == 0):
                # print("Compute")
                return mordred.Calculator(descs=clean_props, ignore_3D=False).pandas(
                    molecules
                )
            else:
                return None
        except Exception as exp:
            print(f"Failed to compute mordred props: + {exp}")
            return None

    def compute_properties_for_mols(self, molecules, as_dataframe: bool = True):
        # try:
        if True:
            mordred_features, mordred_descs = [], None
            mols_df = pd.DataFrame(molecules, columns=["RMol"])
            # print(mols_df.head(2))

            t0 = time()
            rdkit_props = mols_df["RMol"].apply(
                lambda mol: self.compute_rdkit_properties(molecule=mol)
            )
            # print('rdkit_props\n', rdkit_props[0], '\n', rdkit_props.values.tolist())
            t1 = time()
            t_rdkit = t1 - t0
            print(f"RDKIT property calculation: {round(t_rdkit, 3)} seconds.")
            rdkit_props_is_all_none = (
                rdkit_props[0] is None and len(rdkit_props.unique()) == 1
            )
            # print('rdkit_props_is_all_none', rdkit_props_is_all_none)
            # print('self.mordred_descs', self.mordred_descs)
            mordred_props_df = self.compute_mordred_props(molecules, self.mordred_descs)
            t2 = time()
            t_mordred = t2 - t1
            print(f"MORDRED property calculation: {round(t_mordred, 3)} seconds.")
            # print(mordred_props_df is None)
            # if rdkit_props =

            properties = None
            if not (rdkit_props_is_all_none or mordred_props_df is None):
                rdkit_props_df = pd.DataFrame(rdkit_props.values.tolist())
                if as_dataframe:
                    return pd.concat([rdkit_props_df, mordred_props_df], axis=1)
                else:
                    return pd.concat(
                        [rdkit_props_df, mordred_props_df], axis=1
                    ).to_dict(orient="records")

            elif rdkit_props_is_all_none:
                if as_dataframe:
                    return mordred_props_df
                elif not mordred_props_df is None:
                    return mordred_props_df.to_dict(orient="records")
                else:
                    return None

            elif mordred_props_df is None:
                if as_dataframe:
                    if not rdkit_props is None:
                        return pd.DataFrame(rdkit_props.values)
                    else:
                        return None
                else:
                    # print('rdkit_props_df', pd.DataFrame(rdkit_props.values.tolist()).to_dict(orient='records'))
                    # print('rdkit_props', rdkit_props)
                    return rdkit_props.values.tolist()
                    # return pd.DataFrame(rdkit_props.values.tolist()).to_dict(orient='records')

        # except Exception as exp:
        #     print(f"Failed to compute props: {exp}")
        #     return None


ATOMIC_NUM_MAX = 100  ## We only consider atoms with a number of 100 or less (e.g.: 'H':1, 'C':6, 'O':8)
ATOM_FEATURIZER = AtomFeaturizer(
    allowable_sets_one_hot={
        "atomic_num": set(range(1, ATOMIC_NUM_MAX + 1)),
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "formal_charge": {-3, -2, -1, 0, 1, 2, 3},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3", "sp3", "sp3d", "sp3d2"},
        "chiral_tag": {0, 1, 2, 3},
        "is_aromatic": {True, False},
        "is_in_ring": {True, False},
        "is_in_ring_size_5": {True, False},
        "is_in_ring_size_6": {True, False},
    },
    continuous_props=[
        "atomic_mass",
        "atomic_vdw_radius",
        "atomic_covalent_radius",
    ]  # , 'electronegativity'
    # !!!!!!!!!! CHECK lewisacidic/scikit-chem/skchem/core/atom.py on GitHub FOR MANY ATOM PROPERTIES
    # !!! Also check this for additional properties: https://github.com/firmenich/MultiTask-GNN/blob/master/utils/features.py
)

BOND_FEATURIZER = BondFeaturizer(
    allowable_sets_one_hot={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "stereo": {"stereonone, stereoz, stereoe, stereocis, stereotrans"},
        "is_in_ring": {True, False},
        "is_in_ring_size_5": {True, False},
        "is_in_ring_size_6": {True, False},
    }
)


def get_func_groups_pos_from_mol(
    molecule,
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
    as_dict: bool = False,
    label_col: str = "name",
    countUnique: bool = True,
):
    onehot_func_gps = np.zeros(len(df_func_gps), dtype=int)
    try:
        mol = AddHs(molecule)
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            substruct = Chem.MolFromSmarts(smiles)
            match_pos = mol.GetSubstructMatches(substruct)
            if countUnique:
                onehot_func_gps[i] = len(match_pos)
            else:
                onehot_func_gps[i] = int(len(match_pos) > 0)
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()
    except:
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            onehot_func_gps[i] = None
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()


def get_func_groups_pos(
    smiles,
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
    as_dict: bool = False,
    label_col: str = "name",
):
    mol = Chem.MolFromSmiles(smiles)
    mol = AddHs(mol)
    if not mol is None:
        onehot_func_gps = np.zeros(len(df_func_gps), dtype=int)
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            substruct = Chem.MolFromSmarts(smiles)
            match_pos = mol.GetSubstructMatches(substruct)
            onehot_func_gps[i] = len(match_pos)
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()
    else:
        return None


def get_node_dict_from_atom_featurizer(atom_featurizer: AtomFeaturizer):
    feat_mapping = atom_featurizer.features_mapping
    try:
        return {
            feat_mapping["atomic_num"][i]: Chem.Atom(i).GetSymbol()
            if not i == "unk"
            else "unk"
            for i in feat_mapping["atomic_num"]
        }
    except Exception as exp:
        print(exp)
        return None
