#!/usr/bin/env python3
"""Builds California_Housing_Regression.ipynb with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3"}

def md(src):
    return nbf.v4.new_markdown_cell(src.strip())

def code(src):
    return nbf.v4.new_code_cell(src.strip())

nb.cells = [
md("""
# California Housing — Regression, End to End
**Goal:** predict the median house value of a California district from 8 features.
We will do the full ML workflow: explore the data, split it, scale it, train **6 regression models**, compare them, and inspect what the best model learned.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
"""),
md("""
## 1. Load the data
20,640 districts. Target = `MedHouseVal` (median house value in **$100,000s**, so `2.0` means $200,000).
"""),
code("""
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print("Shape:", df.shape)
df.head()
"""),
code("df.describe().T"),
md("""
## 2. Quick EDA — look before you model
"""),
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["MedHouseVal"].hist(bins=50, ax=axes[0])
axes[0].set_title("Target: median house value ($100k)")
axes[0].set_xlabel("$100,000s")
df["MedInc"].hist(bins=50, ax=axes[1])
axes[1].set_title("Feature: median income (x $10k)")
plt.tight_layout(); plt.show()
"""),
code("""
# Which features correlate most with price?
corr = df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values()
corr.plot(kind="barh", figsize=(8, 4), title="Correlation with house price")
plt.tight_layout(); plt.show()
"""),
md("""
## 3. Train / test split
Hide 20% of the districts. Train on 80%, test on the hidden 20% — this is how we catch overfitting.
"""),
code("""
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, " Test:", X_test.shape)
"""),
md("""
## 4. Feature scaling
`Population` is in the thousands, `AveBedrms` is ~1. Standardize so no feature dominates: fit the scaler on **train only**, apply to test.
"""),
code("""
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
"""),
md("""
## 5. Train six regression models
Same data, six algorithms — from the simple straight-line fitter to ensembles.
"""),
code("""
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)":         Ridge(alpha=1.0),
    "Lasso (L1)":         Lasso(alpha=0.1),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    print(name, "trained")
"""),
md("""
## 6. Evaluate — MAE, RMSE, R² on the unseen test set
"""),
code("""
results = []
for name, model in models.items():
    pred = model.predict(X_test_s)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"})
"""),
code("""
results_df.plot(x="Model", y="R2", kind="bar", legend=False, figsize=(9, 4),
                title="R² on test set (higher is better)")
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
"""),
md("""
## 7. What did the best model learn? — feature importance
"""),
code("""
rf = models["Random Forest"]
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", figsize=(8, 4), title="Random Forest feature importance")
plt.tight_layout(); plt.show()
"""),
md("""
## 8. Predict one house yourself
Take the first hidden test district, predict, and convert back to dollars.
"""),
code("""
sample = X_test_s[0].reshape(1, -1)
pred_value = models["Random Forest"].predict(sample)[0]
print(f"Predicted: ${pred_value*100_000:,.0f}   Actual: ${y_test.iloc[0]*100_000:,.0f}")
"""),
md("""
## Takeaways
- **Always start with a simple baseline** (linear regression) — a fancy model is only worth it if it clearly beats it.
- **Random Forest / Gradient Boosting** win on this tabular data (R² ~0.80 vs ~0.60).
- **Income + location** drive the price — the model's feature importance matches real-world intuition.
- **Scale features** for linear models & neural nets; trees don't care.

### Interview questions to practice
1. Why do we split into train and test?
2. MAE or RMSE — when does it matter?
3. Ridge vs Lasso — what's the practical difference?
4. Why fit the scaler on train data only?
5. The forest wins on R² — give one reason you'd still ship linear regression. (Answer: interpretability.)
"""),
]

out = "/home/hatch/workspace/your_files/california-housing-app/California_Housing_Regression.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("notebook written:", out, "| cells:", len(nb.cells))
