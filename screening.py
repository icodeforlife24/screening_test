import joblib
import pandas as pd
import joblib

def load_rf_models():

    return {

        "conductivity":
        joblib.load(
            "models/conductivity_rf.pkl"
        ),

        "stability":
        joblib.load(
            "models/stability_rf.pkl"
        ),

        "viscosity":
        joblib.load(
            "models/viscosity_rf.pkl"
        ),

        "thermal":
        joblib.load(
            "models/thermal_rf.pkl"
        )
    }
from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import Descriptors

descriptor_names = [
    name
    for name, _ in Descriptors._descList
]

def calc_desc(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    desc = {}

    for name, func in Descriptors._descList:

        try:
            desc[name] = func(mol)

        except:
            desc[name] = 0

    return desc
def build_descriptor_dataframe(
    smiles_list
):

    rows = []

    valid_smiles = []

    for smi in smiles_list:

        desc = calc_desc(smi)

        if desc is not None:

            rows.append(desc)
            valid_smiles.append(smi)

    df = pd.DataFrame(rows)

    df = (
        df
        .replace(
            [float("inf"), float("-inf")],
            0
        )
        .fillna(0)
    )

    return df, valid_smiles

def rank_candidates(
    smiles_list,
    descriptor_df,
    rf_models
):

    conductivity = rf_models[
        "conductivity"
    ].predict(
        descriptor_df
    )

    stability = rf_models[
        "stability"
    ].predict(
        descriptor_df
    )

    viscosity = rf_models[
        "viscosity"
    ].predict(
        descriptor_df
    )

    thermal = rf_models[
        "thermal"
    ].predict(
        descriptor_df
    )

    rank_df = pd.DataFrame({

        "SMILES":
        smiles_list,

        "Conductivity":
        conductivity,

        "Stability":
        stability,

        "Viscosity":
        viscosity,

        "Thermal":
        thermal
    })

    rank_df["Score"] = (

        rank_df["Conductivity"]

        + rank_df["Stability"]

        + rank_df["Thermal"]/100

        - rank_df["Viscosity"]/10
    )

    return rank_df.sort_values(
        "Score",
        ascending=False
    )