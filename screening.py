import pandas as pd

def rank_candidates(
    valid,
    candidate_desc,
    rf_models
):

    conductivity = rf_models[
        "conductivity"
    ].predict(candidate_desc)

    stability = rf_models[
        "stability"
    ].predict(candidate_desc)

    viscosity = rf_models[
        "viscosity"
    ].predict(candidate_desc)

    thermal = rf_models[
        "thermal"
    ].predict(candidate_desc)

    df = pd.DataFrame({
        "SMILES": valid,
        "Conductivity": conductivity,
        "Stability": stability,
        "Viscosity": viscosity,
        "Thermal": thermal
    })

    df["Score"] = (
        df["Conductivity"]
        + df["Stability"]
        + df["Thermal"]/100
        - df["Viscosity"]/10
    )

    return df.sort_values(
        "Score",
        ascending=False
    )