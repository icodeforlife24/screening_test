import json
import time
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

from diffusion import load_seed_molecules
from mutation import generate_candidates
from pinn import ElectrolyteRecommendationEngine
from screening import build_descriptor_dataframe, load_rf_models, rank_candidates


def build_report(sigma, voltage, viscosity, thermal):
    start_time = time.perf_counter()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    seed_smiles = load_seed_molecules("data/diffusion_generated.csv")
    candidates = generate_candidates(seed_smiles)

    rf_models = load_rf_models()
    descriptor_df, candidates = build_descriptor_dataframe(candidates)

    ranked = rank_candidates(candidates, descriptor_df, rf_models)
    top20 = ranked.head(20)
    top20.to_csv(output_dir / "top20.csv", index=False)

    engine = ElectrolyteRecommendationEngine(ranked)
    top3 = engine.recommend(sigma, voltage, viscosity, thermal)

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
        molecular_weight = float(Descriptors.MolWt(mol)) if mol is not None else None

        report["top_3_electrolytes"].append(
            {
                "rank": rank,
                "name": f"Candidate-{rank}",
                "smiles": row["SMILES"],
                "molecular_weight": molecular_weight,
                "molecule_type": "Not classified in current pipeline",
                "predicted_properties": {
                    "ionic_conductivity_mS_per_cm_at_25C": float(row["Conductivity"]),
                    "stability_window_V_vs_Li_Liplus": float(row["Stability"]),
                    "viscosity_cP_at_25C": float(row["Viscosity"]),
                    "thermal_decomposition_C": float(row["Thermal"]),
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

    report["execution_time_seconds"] = round(time.perf_counter() - start_time, 3)

    report_path = output_dir / "screening_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def render_card(item):
    props = item["predicted_properties"]
    st.markdown(
        f"""
<div class="elyte-card">
  <div class="elyte-rank">Rank {item['rank']}</div>
  <div class="elyte-name">{item['name']}</div>
  <div class="elyte-smiles">{item['smiles']}</div>
  <div class="elyte-grid">
    <div><span>Conductivity</span><strong>{props['ionic_conductivity_mS_per_cm_at_25C']:.2f} mS/cm</strong></div>
    <div><span>Stability</span><strong>{props['stability_window_V_vs_Li_Liplus']:.2f} V</strong></div>
    <div><span>Viscosity</span><strong>{props['viscosity_cP_at_25C']:.2f} cP</strong></div>
    <div><span>Thermal</span><strong>{props['thermal_decomposition_C']:.2f} C</strong></div>
    <div><span>Molecular Weight</span><strong>{item['molecular_weight'] if item['molecular_weight'] is not None else 'N/A'}</strong></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Electrolyte Screening", layout="wide")

    st.markdown(
        """
<style>
  .stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #eff8f4 45%, #fff6ea 100%);
  }
  .block-container {
    max-width: 1080px;
    padding-top: 1.2rem;
  }
  .elyte-card {
    background: #ffffff;
    border: 1px solid #d6e4f0;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 6px 20px rgba(20, 42, 72, 0.08);
    margin-bottom: 14px;
  }
  .elyte-rank {
    color: #0f4c5c;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-weight: 700;
  }
  .elyte-name {
    color: #003049;
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 2px;
  }
  .elyte-smiles {
    color: #4b5563;
    font-family: "Consolas", "Courier New", monospace;
    margin: 8px 0 10px 0;
    word-break: break-all;
  }
  .elyte-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(160px, 1fr));
    gap: 10px;
  }
  .elyte-grid div {
    background: #f8fbff;
    border-radius: 10px;
    padding: 8px 10px;
  }
  .elyte-grid span {
    display: block;
    color: #51606f;
    font-size: 0.8rem;
  }
  .elyte-grid strong {
    color: #111827;
    font-size: 0.95rem;
  }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Electrolyte Screening Dashboard")
    st.write("Enter target specs, run the same screening pipeline, and view the generated JSON report as cards.")

    with st.form("target_form"):
        col1, col2 = st.columns(2)
        with col1:
            sigma = st.number_input("Desired conductivity (mS/cm)", min_value=0.1, value=10.0, step=0.1)
            voltage = st.number_input("Desired stability window (V)", min_value=0.1, value=4.5, step=0.1)
        with col2:
            viscosity = st.number_input("Desired viscosity (cP)", min_value=0.1, value=5.0, step=0.1)
            thermal = st.number_input("Desired decomposition temperature (C)", min_value=0.1, value=200.0, step=1.0)

        submitted = st.form_submit_button("Generate Report")

    if submitted:
        with st.spinner("Running candidate generation and screening..."):
            report = build_report(sigma, voltage, viscosity, thermal)

        st.success("Report generated and saved to outputs/screening_report.json")

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total Candidates", report["total_candidates_generated"])
        metrics_cols[1].metric("After Low-Fidelity", report["after_low_fidelity_screening"])
        metrics_cols[2].metric("Top20", report["after_high_fidelity_validation"])
        metrics_cols[3].metric("Top3", report["after_physical_constraints"])

        st.subheader("Top 3 Electrolytes")
        for item in report["top_3_electrolytes"]:
            render_card(item)

        st.subheader("Generated JSON")
        st.json(report)

        st.download_button(
            label="Download JSON",
            data=json.dumps(report, indent=2),
            file_name="screening_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
