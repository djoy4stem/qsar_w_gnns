import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AddHs,
    Draw,
    AllChem,
    Crippen,
    Lipinski,
    QED,
    Descriptors,
    GraphDescriptors,
    Fragments,
    MolToSmiles,
    GetPeriodicTable,
    MACCSkeys,
)
from rdkit.Chem import rdMolDescriptors as rdmdesc
from rdkit.Chem import rdPartialCharges as rdpart
from typing import List, Any, Union
from lib import utilities
import re
import mordred
from mordred import descriptors, Calculator
from time import time
import warnings
import ast

RDLogger.DisableLog("rdApp.*")


RDKIT_FRAGS = [
    [frag_name, Chem.MolToSmarts(eval(f"Fragments.{frag_name}").__defaults__[1])]
    for frag_name in dir(Fragments)
    if not re.match("^fr", frag_name) is None
]


def get_maccs_smarts():
    patts = MACCSkeys.smartsPatts
    # return [[f"maccs_{k}",patts[k][0]]  for k in patts if not patts[k][0] == '?'][:] # Remove the first pattern ('?') because of parsing issues.
    return [
        [f"maccs_{k}", patts[k][0]] for k in patts if not k in [1, 125, 166]
    ]  # Remove the first pattern ('?') because of parsing issues.


def get_maccs_substructures():
    """
    Returns a list of (bit, substructure) tuples for MACCS keys.
    Each bit corresponds to a substructure pattern.
    """
    # maccs_bits = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles("CC"))
    maccs_bits = get_maccs_smarts()
    substructures = []

    # Map each MACCS bit to a substructure. Note that some bits do not directly
    # correspond to specific substructures and may return None.
    # for bit in range(1, maccs_bits.GetNumBits()):
    for item in range(len(maccs_bits)):
        # print("bit", bit)
        # print(bit, MACCSkeys.smartsPatts[bit])
        bit = maccs_bits[item][0]
        substructure = maccs_bits[item][1]
        if substructure:
            # print(substructure)
            substructures.append(
                (bit, Chem.MolFromSmarts(substructure), (bit, item))
            )  # The last will be helpful, as some maccs keys were removed.
        else:
            substructures.append((bit, None, (bit, item)))
    # print(len(substructures))
    return substructures


def get_substructures(func_groups: Union[List, str]):
    substructures = list(
        map(lambda item: [item[0], Chem.MolFromSmarts(item[1])], func_groups)
    )
    if func_groups == "maccs":
        return get_maccs_substructures()
    else:
        for i in range(len(substructures)):
            substructures[i].append([substructures[i][0], i])
        return substructures


MACCS_FP = pd.DataFrame(get_maccs_smarts(), columns=["name", "SMARTS"])
DF_FUNC_GRPS_RDKIT = pd.DataFrame(RDKIT_FRAGS, columns=["name", "SMARTS"])

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
        [
            "dialkylether",
            "[#6;A;X4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])][#8;X2][#6;A;X4;!$(C([OX2])[O,S,#7,#15])]",
        ],
        [
            "diarylether",
            "[$([cX3](:[!#1;*]):[!#1;*]),$([cX2+](:[!#1;*]):[!#1;*])]-[#8]-[$([cX3](:[!#1;*]):[!#1;*]),$([cX2+](:[!#1;*]):[!#1;*])]",
        ],
        [
            "dialkylthioether",
            "[#6;A;X4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])][#16;X2][#6;A;X4;!$(C([OX2])[O,S,#7,#15])]",
        ],
        ["diarylthioether", "[#6;a]-[#16;X2]-[#6;a]"],
        ["lactone", "[#6][#6X3R](=[OX1])[#8X2][#6;!$(C=[O,N,S])]"],
        [
            "lactam",
            "	[#6R][#6X3R](=[OX1])[#7X3;$([H1][#6;!$(C=[O,N,S])]),$([H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        ["alcohol", "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"],
        ["arom_alcohol", "[OX2H][c]"],
        ["arom_ether", "[OX2]([C,c])[c]"],
        ["amine", "[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]"],
        ["dialkylarylamine", "[#6;A;X4][#7;v3X3](-[#6;a])[#6;A;X4]"],
        ["alkyldiarylamine", "[#6;A;X4][#7;v3X3;!R](-[#6;a])-[#6;a]"],
        ["arylhalide", "[Cl,F,I,Br][c]"],
        ["alkylhalide", "[Cl,F,I,Br][CX4]"],
        ["acylhalide", "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[Cl,F,I,Br]"],
        [
            "aldimine",
            "[H][#6](-[#6])!@=[#7;A;X2;$([N][#6]),$(n);!$([N][CX3]=[#7,#8,#15,#16])]",
        ],
        ["aldoxime", "[H][#8]!@\[#7]!@=[#6](/[H])-[#6]"],
        ["aldamine", "[H]\[#6](-[#6])!@=[#7]!@\[#7]!@=[#6](/[H])-[#6]"],
        ["pyran", "[$(C1OC=CC=C1),$(C1CC=CCO1),$(C1C=COC=C1)]"],
        ["pyrazole", "c1cnnc1"],
        ["pyridine", "c1ccncc1"],
        ["pyrimidine", "c1cncnc1"],
        ["purine", "c1nc2cncnc2n1"],
        ["n_arylamide", "[H][#7;X3R0](-[#6;a])-[#6;X3R0]([#6,#1;A])=[O;X1]"],
        ["n_acyl_piperidine", "[#6]!@-[#6](=O)!@-[#7]-1-[#6]-[#6]-[#6]-[#6]-[#6]-1"],
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
        ["2,4-disubstituted-1,3-oxazole", "[H]c1oc(!@-[!#1;*])nc1!@-[!#1;*]"],
        ["3-aroylfuran", "[!#1;A;a]-[#6](=O)!@-c1ccoc1"],
        ["3-aroylthiophene", "[!#1;A;a]-[#6](=O)!@-c1ccsc1"],
        ["cepham", "O=C1CC2SCCCN12"],
        ["chloroalkene", "[Cl;X1][#6;A;X3]=[#6;A;X3]"],
        ["clavam", "O=C1CC2OCCN12"],
        ["cyanate", "OC#N"],
        ["cyanate_ester", "COC#N"],
        [
            "cyclic_ketone",
            "[$([#6;R1]-[#6;R1](-[#6;R1])=[O;v2]),$([#6;R1][#6;R1]([#6;R1])=[O;v2])]",
        ],
        ["delta_lactam", "O=C1CCCCN1"],
        ["disulfide", "[#6]-[#16;X2]!@-[#16;X2]-[#6]"],
        ["morpholine", "C1COCCN1"],
        ["n_acetylarylamine", "[H][#7;X3R0](-[#6;a])-[#6;R0](-[#6;H3X4])=O"],
        ["n_acyl-piperidine", "[#6]!@-[#6](=O)!@-[#7]-1-[#6]-[#6]-[#6]-[#6]-[#6]-1"],
        ["n_acylimidazole", "[#6,#1]-[#6](=[O;X1])-n1ccnc1"],
        ["n_substituted_pyrrole", "[!#1;*]!@-n1cccc1"],
        ["nitroaromatic_compound", "[$([#6;a]-[#7+](-[#8-])=O),$([#6;a]N(=O)=O)]"],
        ["nitrosourea", "[#7]!@-[#6](!@=O)!@-[#7]!@-[#7]!@=O"],
        ["oxane", "C1CCOCC1"],
        ["oxaziridine", "C1NO1"],
        ["penam", "O=C1CC2SCCN12"],
        ["penem", "O=C1CC2SC=CN12"],
        ["pubchem_561", "O=S-C-C"],
        ["pubchem_577", "O-S-C:C"],
        ["pubchem_610", "O-N-C-C"],
        ["pubchem_817", "Cc1ccc(C)cc1"],
        ["pubchem_720", "Oc1ccc(S)cc1"],
        ["pubchem_879", "Brc1c(Br)cccc1"],
        ["pubchem_970", "SC1C(Br)CCC1"],
        ["sp3_nitrogen", "[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]"],
        ["unbranched_alkane", "[R0;D2][R0;D2][R0;D2][R0;D2]"],
        ["unbranched_chain", "[R0;D2]~[R0;D2]~[R0;D2]~[R0;D2]"],
        # ["para_subs_ring", "*-!:aaaa-!:*"],
        # ["meta_subs_ring", "*-!:aaa-!:*"],
        # ["ortho_subs_ring", "*-!:aa-!:*"],
        ["azide_ion", "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]"],
        ["N#S", "N#S"],
        ["N=S", "N=S"],
        ["S=S", "S=S"],
        ["C#S", "C#S"],
        ["trihalomethyl", "[$([#6;X4]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I])]"],
        ["organohalogen", "[#6;+0]!@-[F,Cl,Br,I]"],
        ["n_substituted_imidazole", "[!#1;*]n1ccnc1"],
        ["n_arylpiperazine", "[#6;a]!@-[#7]-1-[#6]-[#6]-[#7]-[#6]-[#6]-1"],
        ["n_alkylpiperazine", "[#6;A;X4][#7]-1-[#6]-[#6]-[#7]-[#6]-[#6]-1"],
        [
            "michael_acceptor",
            "[CX3]=[CX3][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-])]",
        ],
        ["acylsulfonic_acid_deriv", "[!#1!#6][S;v6](=O)(=O)[#6;R0]=O"],
        ["alkanesulfonic_acid_deriv", "[#6;A;X4]S([#8;X2])([#8;X2])=O"],
        ["alkoxide", "[#6;A;X4][#8;D1R0-]"],
        [
            "alkyl_arylether",
            "[#6;a]-[#8;X2][#6;A;X4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]",
        ],
        [
            "alkyl_arylthioether",
            "[#6;a]-[#16;X2][#6;A;X4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]",
        ],
        [
            "alkylthiol",
            "[#1,C,$([cX3](:[!#1;*]):[!#1;*]),$([cX2+](:[!#1;*]):[!#1;*])]-,=[#6]-[#16][H]",
        ],
        ["allyl_alcohol", "[H]!@-[#8]!@-[#6]!@-[#6]!@=[#6]"],
        ["alpha,beta_enoate_ester", "[#6;!$(C=[O,N,S])][#8;A;D2][#6](=O)-[#6]=[#6;X3]"],
        ["alpha_diketone", "[#6]!@-[#6](!@=O)!@-[#6](!@-[#6])!@=O"],
        ["alpha_haloketone", "[#6]-[#6;X3](=[O;X1])-[#6]-[F,Cl,Br,I]"],
        [
            "alpha_hydroxyacid",
            "[H][#8;v2]-[#6](-[*;#1,C,$([cX3](:[!#1;*]):[!#1;*]),$([cX2+](:[!#1;*]):[!#1;*])])-[#6](=O)-[#8][H]",
        ],
        ["alpha_ketoaldehyde", "[H][#6;R0](=[O;R0])-[#6;R0](-[#6])=[O;R0]"],
        [
            "carboxamide",
            "[#7;A;X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])][#6](-[#6,#1])=[O;X1]",
        ],
        ["carboximidamide", "[#6,#1;A][#7]=[#6]-[#7]([#6,#1;A])[#6,#1;A]"],
        ["carboxylic_thioester", "[#6]-[#16;X2]-[#6;X3](-[#6])=O"],
        [
            "carboxylic_acid_imide",
            "[#6,#1]-[#7;X3](-[#6;X3](-[#6,#1])=[O;X1])-[#6;X3](-[#6,#1])=[O;X1]",
        ],
        [
            "carboxylic_acid_imide_n_unsub",
            "[H][#7;X3](-[#6;X3](-[#6,#1])=[O;X1])-[#6;X3](-[#6,#1])=[O;X1]",
        ],
        [
            "carboxylic_acid_imide_n_sub",
            "[#6]-[#7;X3](-[#6;X3](-[#6,#1])=[O;X1])-[#6;X3](-[#6,#1])=[O;X1]",
        ],
        ["imidazole", "c1cncn1"],
        ["imine", "[#6,#1;A][#7;A;X2;!$(N~C~[!#1#6])]=[#6;A;X3,A]"],
        ["indole", "c1cc2ccccc2n1"],
        [
            "phenylketone",
            "[#6;X4]-[#6;X3](=O)!@-[c;R1]1[c;R1][c;R1][c;R1][c;R1][c;R1]1",
        ],
        ["acryloyl", "[#6;!R]-[#6](=O)-[#6;!R]=[#6;!R]"],
        [
            "alpha,beta_unsat_carbonyl",
            "[$([*;#1,N,O,S,X,C]-[#6;R0](=O)[C;R0]!@#C),$([*;#1,N,O,S,X,C]-[#6;R0](=O)-[#6;R0]!@=[#6;X3])]",
        ],
        ["alpha,beta-unsat_ketone", "[#6]-[#6](=O)-[#6;R0]=[#6;R0]"],
        ["alpha_hydroxyketone", "[H][#8;!R]-[#6;!R]-[#6](-[#6;!R])=O"],
        ["alpha_keto-acid", "[#6]!@-[#6](=O)!@-[#6](!@-[#8])=O"],
        ["alpha_hydroxy_ketone", "[OX2H1]CC(C)=O"],
        ["amidoxime", "[#8;X2H1]\[#7]=[#6](\[#6])-[#7]([#6,#1;A])[#6,#1;A]"],
        [
            "aminal",
            "[*;CX4,#1]-[#7](-[*;CX4,#1])C([*;CX4,#1])([*;CX4,#1])[#7](-[*;CX4,#1])-[*;CX4,#1]",
        ],
        ["isopropyl", "[#6;A;H3X4][#6;H1X4]([#6;A;H3X4])-[#7;X3]"],
        [
            "enolether",
            "[#6;!$(C=[N,O,S])]-[#8;X2][#6;A;X3;$([H0][#6]),$([H1])]=[#6;A;X3]",
        ],
        [
            "enolester",
            "[#6;X3;!$(C[OX2H])]=[#6;X3;$([#6][#6]),$([H1])]-[#8;X2][#6;A;X3]=[O;X1]",
        ],
        ["enolester_epoxide", "[#6]-[#6]-1-[#8]-[#6]-1-[#8]-[#6]([#6,#1;A])=O"],
        ["enolate", "[#8;v1-]!@\[#6]([#6,#1;A])!@=[#6](\[#6,#1;A])[#6,#1;A]"],
        ["enone", "[#6;!R]-[#6](=[O;!R])-[#6;!R]=[#6;!R]"],
        ["organoboron", "[#6][BX3]"],
        [
            "organo_sulfenic_acid_deriv",
            "[$([#6]-[#16;v2X2][!#1!#6;N,O,X]),$([H][S;H1X4]([#6])=[O;X1])]",
        ],
        ["organosulfimide", "[#6;+0]-[#7]=S(=O)=O"],
        [
            "organosulfonic_acid_deriv",
            "[#6;!$(C=[O,N,S])]-[#8][S;v6]([#6;!$(C=[O,N,S])])(=[O;X1])=[O;X1]",
        ],
        [
            "sulicnic_acid_deriv",
            "[C,$([cX3](:[!#1;*]):[!#1;*]),$([cX2+](:[!#1;*]):[!#1;*])][S;v4X3]([!#1!#6;O,N,X])=O",
        ],
        ["sulfinyl_halide", "[#6][S;v4X3]([F,Cl,Br,I])=O"],
        [
            "sulfuric_acid_diamide",
            "[#7;A;X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])][S;X4]([#7;A;X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])(=[O;X1])=[O;X1]",
        ],
        [
            "sulfuric_acid_monoester",
            "[#6]-[#8;X2][S;X4]([#8;A;X2H1,X1-])(=[O;X1])=[O;X1]",
        ],
        ["sulfuric_acid_diester", "[#6]-[#8;X2][S;X4](=[O;X1])(=[O;X1])[#8;X2]-[#6]"],
        ["on_c_c", "[#8]~[#7](~[#6])~[#6]"],
        ## https://github.com/openbabel/openbabel/blob/master/data/SMARTS_InteLigand.txt ##
        ["epoxide", "[OX2r3]1[#6r3][#6r3]1"],
        ["spiro", "[D4R;$(*(@*)(@*)(@*)@*)]"],
        [
            "n_hetero_imide",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#7X3H0]([!#6])[#6X3;$([H0][#6]),$([H1])](=[OX1])",
        ],
        [
            "alkyl_imide",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#7X3H0]([#6])[#6X3;$([H0][#6]),$([H1])](=[OX1])",
        ],
        [
            "thioamide",
            "[$([CX3;!R][#6]),$([CX3H;!R])](=[SX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        ["prim_amide", "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[NX3H2]"],
        ["sec_amide", "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H1][#6;!$(C=[O,N,S])]"],
        [
            "ter_amide",
            " [CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])]",
        ],
        [
            "carbothioic_s_ester",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[SX2][#6;!$(C=[O,N,S])]",
        ],
        ["carbothioic_s_lactone", "[#6][#6X3R](=[OX1])[#16X2][#6;!$(C=[O,N,S])]"],
        [
            "carbothioic_o_ester",
            "[CX3;$([H0][#6]),$([H1])](=[SX1])[OX2][#6;!$(C=[O,N,S])]",
        ],
        ["carbothioic_o_lactone", "[#6][#6X3R](=[SX1])[#8X2][#6;!$(C=[O,N,S])]"],
        ["carbothioic_halide", "[CX3;$([H0][#6]),$([H1])](=[SX1])[FX1,ClX1,BrX1,IX1]"],
        ["carbodithioic_acid", "[CX3;!R;$([C][#6]),$([CH]);$([C](=[SX1])[SX2H])]"],
        [
            "carbodithioic_ester",
            "[CX3;!R;$([C][#6]),$([CH]);$([C](=[SX1])[SX2][#6;!$(C=[O,N,S])])]",
        ],
        ["carbodithiolactone", "[#6][#6X3R](=[SX1])[#16X2][#6;!$(C=[O,N,S])]"],
        [
            "amide",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        [
            "_1,5_tautomerizable",
            "[$([#7X2,OX1,SX1]=,:**=,:*[!H0;!$([a;!n])]),$([#7X3,OX2,SX2;!H0]*=**=*),$([#7X3,OX2,SX2;!H0]*=,:**:n)]",
        ],
        [
            "ch_acidic",
            "[$([CX4;!$([H0]);!$(C[!#6;!$([P,S]=O);!$(N(~O)~O)])][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])]),$([CX4;!$([H0])]1[CX3]=[CX3][CX3]=[CX3]1)]",
        ],
        [
            "ch_acidic_strong",
            "[CX4;!$([H0]);!$(C[!#6;!$([P,S]=O);!$(N(~O)~O)])]([$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])])[$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])]",
        ],
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

    def featurizer_to_values(self) -> pd.Series:
        pos, values, f_types = [], [], []
        for prop in self.features_mapping.items():
            # print("prop", prop)

            if isinstance(prop[1], int):
                values.append(None)
                pos.append(prop[1])
                f_types.append(prop[0])
            elif isinstance(prop[1], dict):
                # print("prop[1]",  prop[1].items())
                for p in prop[1].items():
                    # print("p", p)
                    values.append(p[0])
                    pos.append(p[1])
                    f_types.append(prop[0])
            else:
                raise ValueError(f"prop[1] is not of type 'int' or 'dict'. {prop[1]}")

        pos_to_feats = pd.DataFrame()
        pos_to_feats["value"] = values
        pos_to_feats["f_type"] = f_types
        pos_to_feats.index = pos

        return pos_to_feats


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets_one_hot, continuous_props=None):
        super().__init__(allowable_sets_one_hot, continuous_props)
        self.atomic_feats_pos_values = self.featurizer_to_values()

    ## To Do
    ## Think of adding an attribute for enhanced node represention
    # susing specific functional groups. For instance, set
    # attribute frags_for_nodes = {'frag1':['C#S'], 'primary_alcol':'smarts1',etc...}
    # The encoding will return a vector of size
    # len(frags_for_nodes) with values 0 or 1 to
    # explain whether the atom is part of the
    # corresponding fragment. The 'frags_for_nodes'
    # key must be added to the features mapping

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
        return atom.GetChiralTag().name.lower()

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

    def is_h_acceptor(self, atom):
        """
        Is an H acceptor?
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        m = atom.GetOwningMol()
        idx = atom.GetIdx()
        return idx in [i[0] for i in Lipinski._HAcceptors(m)]

    def is_h_donor(self, atom):
        """
        Is an H donor?
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        m = atom.GetOwningMol()
        idx = atom.GetIdx()
        return idx in [i[0] for i in Lipinski._HDonors(m)]

    def is_heteroatom(self, atom):
        """
        Is an heteroatom?
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        m = atom.GetOwningMol()
        idx = atom.GetIdx()
        return idx in [i[0] for i in Lipinski._Heteroatoms(m)]

    def num_implicit_hydrogens(self, atom):
        """
        Number of implicit hydrogens
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        return atom.GetNumImplicitHs()

    def num_explicit_hydrogens(self, atom):
        """
        Number of implicit hydrogens
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        return atom.GetNumExplicitHs()

    def crippen_log_p_contrib(self, atom):
        """
        Hacky way of getting logP contribution.
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        idx = atom.GetIdx()
        m = atom.GetOwningMol()
        if not m.GetPropsAsDict().get("CLogPAtomContribs", None) is None:
            return ast.literal_eval(m.GetProp("CLogPAtomContribs"))[idx][0]
        else:
            warnings.warn(
                "Owning mol has not calculated CLogPAtomContribs. Calculating now..."
            )
            m.SetProp("CLogPAtomContribs", str(Crippen._GetAtomContribs(m)))
            return ast.literal_eval(m.GetProp("CLogPAtomContribs"))[idx][0]

    def crippen_molar_refractivity_contrib(self, atom):
        """
        Hacky way of getting molar refractivity contribution.

        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        idx = atom.GetIdx()
        m = atom.GetOwningMol()
        if not m.GetPropsAsDict().get("CLogPAtomContribs", None) is None:
            return ast.literal_eval(m.GetProp("CLogPAtomContribs"))[idx][1]
        else:
            warnings.warn(
                "Owning mol has not calculated CLogPAtomContribs. Calculating now..."
            )
            m.SetProp("CLogPAtomContribs", str(Crippen._GetAtomContribs(m)))
            return ast.literal_eval(m.GetProp("CLogPAtomContribs"))[idx][1]

    def tpsa_contrib(self, atom):
        """
        Hacky way of getting total polar surface area contribution.

        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        idx = atom.GetIdx()
        m = atom.GetOwningMol()

        if not m.GetPropsAsDict().get("TPSAContribs", None) is None:
            return ast.literal_eval(m.GetProp("TPSAContribs"))[idx]
        else:
            warnings.warn(
                "Owning mol has not calculated AtomContribs. Calculating now..."
            )
            m.SetProp("TPSAContribs", str(rdmdesc._CalcTPSAContribs(m)))
            return ast.literal_eval(m.GetProp("TPSAContribs"))[idx]

    def labute_asa_contrib(self, atom):
        """
        Hacky way of getting accessible surface area contribution.
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        idx = atom.GetIdx()
        m = atom.GetOwningMol()

        if not m.GetPropsAsDict().get("LabuteASAContribs", None) is None:
            return ast.literal_eval(m.GetProp("LabuteASAContribs"))[0][idx]
        else:
            warnings.warn(
                "Owning mol has not calculated AtomContribs. Calculating now..."
            )
            lacontribs = rdmdesc._CalcLabuteASAContribs(m)
            # print(list(lacontribs[0]))
            m.SetProp("LabuteASAContribs", str([list(lacontribs[0]), lacontribs[1]]))
            return ast.literal_eval(m.GetProp("LabuteASAContribs"))[0][idx]

    def gasteiger_charge(self, atom, force_calc=False):
        """
        Hacky way of getting gasteiger charge
        From scikit-learn
        https://github.com/lewisacidic/scikit-chem/blob/master/skchem/features/atom.py
        """

        if atom.GetPropsAsDict().get("_GasteigerCharge", None) is None or force_calc:
            m = atom.GetOwningMol()
            rdpart.ComputeGasteigerCharges(m)
            return float(atom.GetProp("_GasteigerCharge"))
        else:
            return atom.GetProp("_GasteigerCharge")


class BondFeaturizer(Featurizer):
    ## To Do
    ## Think of adding an attribute for enhanced bond represention
    # susing specific functional groups. For instance, set
    # attribute frags_for_nodes = {'frag1':['C#S'], 'primary_alcol':'smarts1',etc...}
    # The encoding will return a vector of size
    # len(frags_for_nodes) with values 0 or 1 to
    # explain whether the bond is part of the
    # corresponding fragment. The 'frags_for_nodes'
    # key must be added to the features mapping

    def __init__(self, allowable_sets_one_hot):
        # allowable_sets_one_hot["bond_none"]={}
        super().__init__(allowable_sets_one_hot)
        self.dim += 1
        self.bond_feats_pos_values = self.featurizer_to_values()

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
        df_func_gps: pd.DataFrame = DF_FUNC_GRPS_RDKIT,
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
            t0 = time()
            # print("Performing structure search...")
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
            # print(f"Structure search took {time() - t0} sec...")
            # print(f"properties = {properties}")
            if bool(properties):
                if not as_dict:
                    prop_values = []

                    for i in list(properties.values()):
                        if isinstance(i, (float, int, bool)):
                            prop_values.append(i)
                        elif isinstance(i, (list, tuple, set)):
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
                    if isinstance(i, (list, tuple, set)):
                        prop_values.append(i)
                    elif isinstance(i, (list, tuple, set)):
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
        try:
            # if True:
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
            # print(f"RDKIT property calculation: {round(t_rdkit, 3)} seconds.")
            rdkit_props_is_all_none = (
                rdkit_props[0] is None and len(rdkit_props.unique()) == 1
            )
            # print('rdkit_props_is_all_none', rdkit_props_is_all_none)

            mordred_props_df = None
            if not (self.mordred_descs is None or len(self.mordred_descs) == 0):
                # print('self.mordred_descs', self.mordred_descs)
                mordred_props_df = self.compute_mordred_props(
                    molecules, self.mordred_descs
                )
                t2 = time()
                t_mordred = t2 - t1
                # print(f"MORDRED property calculation: {round(t_mordred, 3)} seconds.")
                # print(mordred_props_df is None)

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

        except Exception as exp:
            print(f"Failed to compute props: {exp}")
            return None


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
        "is_h_acceptor": {True, False},
        "is_h_donor": {True, False},
        "is_heteroatom": {True, False},
    },
    continuous_props=[
        "atomic_mass",
        "atomic_vdw_radius",
        "atomic_covalent_radius",
        "num_implicit_hydrogens",
        "num_explicit_hydrogens",
        "crippen_log_p_contrib",
        "crippen_molar_refractivity_contrib",
        "tpsa_contrib",
        "labute_asa_contrib",
        "gasteiger_charge",
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
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS_RDKIT,
    as_dict: bool = False,
    label_col: str = "name",
    countUnique: bool = True,
):
    onehot_func_gps = np.zeros(len(df_func_gps), dtype=int)
    try:
        mol = AddHs(molecule)
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            # print(df_func_gps.iloc[i]['name'], smiles)
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
            print(i)
            onehot_func_gps[i] = None
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()


def get_func_groups_pos(
    smiles,
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS_RDKIT,
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
