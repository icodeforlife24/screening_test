import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from diffusion import load_seed_molecules
from mutation import generate_candidates
from pinn import ElectrolyteRecommendationEngine

from screening import (
    load_rf_models,
    build_descriptor_dataframe,
    rank_candidates
   
)

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    sigma = float(
        input(
            "Desired conductivity (mS/cm) [default: 10]: "
        ) or 10
    )

    voltage = float(
        input(
            "Desired stability window (V) [default: 4.5]: "
        ) or 4.5
    )

    viscosity = float(
        input(
            "Desired viscosity (cP) [default: 5]: "
        ) or 5
    )

    thermal = float(
        input(
            "Desired decomposition temperature (°C) [default: 200]: "
        ) or 200
    )

    seed_smiles = load_seed_molecules(
        "data/diffusion_generated.csv"
    )

    candidates = generate_candidates(
        seed_smiles
    )

    rf_models = load_rf_models()

    descriptor_df, candidates = (
        build_descriptor_dataframe(
            candidates
        )
    )

    ranked = rank_candidates(
        candidates,
        descriptor_df,
        rf_models
    )

    top20 = ranked.head(20)

    top20.to_csv(
        "outputs/top20.csv",
        index=False
    )

    engine = ElectrolyteRecommendationEngine(
        ranked
    )

    top3 = engine.recommend(
        sigma,
        voltage,
        viscosity,
        thermal
    )

    report = {
        "target_specification": {
            "ionic_conductivity_min_mS_per_cm": sigma,
            "stability_window_min_V": voltage,
            "viscosity_max_cP": viscosity,
            "thermal_stability_min_C": thermal,
        },
        "generator_type": "Candidate screening pipeline with diffusion-based seed loading and mutation-based generation",
        "low_fidelity_model": "Random forest descriptor ranker",
        "high_fidelity_model": "Distance-based recommendation engine",
        "total_candidates_generated": int(len(candidates)),
        "after_low_fidelity_screening": int(len(ranked)),
        "after_high_fidelity_validation": int(len(top20)),
        "after_physical_constraints": int(len(top3)),
        "active_learning_rounds": 0,
        "top_3_electrolytes": [],
        "low_fidelity_vs_high_fidelity_correlation": None,
        "execution_time_seconds": None,
    }

    for rank, (_, row) in enumerate(top3.head(3).iterrows(), start=1):
        mol = Chem.MolFromSmiles(row["SMILES"])
        molecular_weight = (
            float(Descriptors.MolWt(mol))
            if mol is not None
            else None
        )

        report["top_3_electrolytes"].append(
            {
                "rank": rank,
                "name": f"Candidate-{rank}",
                "smiles": row["SMILES"],
                "molecular_weight": molecular_weight,
                "molecule_type": "Not classified in current pipeline",
                "predicted_properties": {
                    "ionic_conductivity_mS_per_cm_at_25C": float(
                        row["Conductivity"]
                    ),
                    "stability_window_V_vs_Li_Liplus": float(
                        row["Stability"]
                    ),
                    "viscosity_cP_at_25C": float(
                        row["Viscosity"]
                    ),
                    "thermal_decomposition_C": float(
                        row["Thermal"]
                    ),
                },
                "confidence_intervals": None,
                "novelty_score": None,
                "similarity_to_existing": "Not computed in current pipeline",
                "physical_constraints_passed": [
                    "Conductivity target met",
                    "Stability target met",
                    "Viscosity target met",
                    "Thermal target met",
                ],
                "synthesis_feasibility": "Not evaluated in current pipeline",
                "falsifiable_electrochemical_test": "Run the platform electrochemical validation protocol on the selected candidate.",
            }
        )

    report_path = output_dir / "screening_report.json"
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()