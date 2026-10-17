# California Housing Price Predictor

Predicting California median house values with regression — one project, three ways to explore it.

## What's inside

| Path | What it is |
|---|---|
| `California_Housing_Regression.ipynb` | Full Jupyter walkthrough: EDA, train/test split, scaling, 6 regression models compared, feature importance |
| `streamlit/` | Interactive Streamlit app — sliders for all 8 features, live prediction, model comparison tab. Deploy free on [Streamlit Cloud](https://share.streamlit.io) (see `streamlit/README.md`) |
| `netlify/` | Static frontend — the trained Ridge model exported to `model.json`, predictions run in JavaScript. Deploy free on [Netlify](https://app.netlify.com/drop) (see `netlify/README.md`) |

## The model

Ridge regression (L2, α=1.0) trained on 16,512 California districts, tested on 4,128 unseen districts:

- R² ≈ 0.58 · MAE ≈ 0.53 · RMSE ≈ 0.75 (target in $100,000s)

Try the ensembles in the notebook — Random Forest reaches R² ≈ 0.80 on this data.

## Run locally

```bash
pip install scikit-learn pandas numpy matplotlib streamlit
jupyter notebook California_Housing_Regression.ipynb
# or
streamlit run streamlit/app.py
```
