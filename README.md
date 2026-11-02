# DeGroot Learning Model - Streamlit App

Interactive simulation of DeGroot opinion dynamics, `x(t+1) = W x(t)`:
editable trust matrix, scenario presets, trust heatmap, network graph,
eigenvalue plot, convergence chart and a playable animation.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Dependencies installed by Streamlit Cloud |
| `.streamlit/config.toml` | Theme / server settings |
| `.gitignore` | Keeps junk out of the repo |
