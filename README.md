# Screening Test

A small electrolyte screening project that generates candidate molecules, ranks them with low-fidelity models, and produces a final recommendation report.

## What it does

- Loads seed SMILES from `data/diffusion_generated.csv`
- Generates additional candidate molecules
- Builds descriptors and ranks candidates
- Produces a JSON report with the top results
- Saves output files in `outputs/`

## Main files

- `main.py` - command-line pipeline
- `streamlit_app.py` - simple Streamlit dashboard
- `diffusion.py` - candidate generation
- `mutation.py` - more varying candidate generation
- `screening.py` - descriptor and ranking logic
- `pinn.py` - recommendation step

## How to run

Command line:

```powershell
python main.py
```

Streamlit app:

```powershell
streamlit run streamlit_app.py
```

## Output

Results are written to the `outputs/` folder, including:

- `top20.csv`
- `screening_report.json`
