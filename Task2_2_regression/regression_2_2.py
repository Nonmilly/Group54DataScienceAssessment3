from pathlib import Path

"""
Regression 2.2 — Team-Level Goal Scoring
Multiple Linear Regression vs Poisson Regression

Input workbook (must be beside this script):
    regression_2_2_team_match_208_with_scores.xlsx

Required sheet:
    Team_Match_208

Run from PowerShell:
    .\.venv\Scripts\python.exe .\regression_2_2.py

This model predicts the number of goals scored by a team in a FIFA World Cup 2026 match using eight pre-match explanatory variables. Four predictors are shared with Regression 2.1, and four are additional predictors. The analysis compares Multiple Linear Regression and Poisson Regression.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
INPUT = BASE / "regression_2_2_team_match_208_with_scores.xlsx"
OUT = BASE / "regression_2_2_outputs"
OUT.mkdir(exist_ok=True)

TARGET = "GoalScored"
PREDICTORS = [
    "RankPointsDiff_TeamVsOpponent",
"HostAdvantage_TeamVsOpponent",
"FormGoalsForDiff_TeamVsOpponent",
"FormGoalsAgstDiff_TeamVsOpponent",
"SquadAvgCapsDiff_TeamVsOpponent",
"SquadAvgIntlGoalsDiff_TeamVsOpponent",
"SquadAvgHeightDiff_TeamVsOpponent",
"OverseasClubShareDiff_TeamVsOpponent",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

print("=" * 60)
print("REGRESSION 2.2 — GOAL SCORING ANALYSIS")
print("=" * 60)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Excel workbook not found: {INPUT}\n"
        "Place the workbook in the same folder as this script."
    )

df = pd.read_excel(INPUT, sheet_name="Model_Ready_208")

required_columns = ["MatchID", TARGET] + PREDICTORS
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

data = df[required_columns].copy()
for col in [TARGET] + PREDICTORS:
    data[col] = pd.to_numeric(data[col], errors="coerce")

if data[required_columns].isna().any().any():
    missing_counts = data[required_columns].isna().sum()
    raise ValueError(
        "Missing or invalid values found:\n"
        + missing_counts[missing_counts > 0].to_string()
    )

if len(data) != 208 or data["MatchID"].nunique() != 104:
    raise ValueError(
        f"Expected 208 team-match rows from 104 matches, but found "
        f"{len(data)} rows and {data['MatchID'].nunique()} matches."
    )

if not (data.groupby("MatchID").size() == 2).all():
    raise ValueError("Each MatchID must contain exactly two team records.")

if (data[TARGET] < 0).any():
    raise ValueError("GoalScored cannot contain negative values.")

if not np.allclose(data[TARGET], data[TARGET].round()):
    raise ValueError("GoalScored should contain whole-number counts.")

print(f"\nRows loaded: {len(data)}")
print(f"Unique matches: {data['MatchID'].nunique()}")
print(f"Missing values: {data[required_columns].isna().sum().sum()}")

# --------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print(data[[TARGET] + PREDICTORS].describe().round(4))
data[[TARGET] + PREDICTORS].describe().to_csv(
    OUT / "descriptive_statistics.csv"
)

# --------------------------------------------------
# 4. FEATURES, TARGET AND MATCH-GROUPED SPLIT
# --------------------------------------------------

X = data[PREDICTORS].astype(float)
y = data[TARGET].astype(float)
groups = data["MatchID"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
train_i, test_i = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_i]
X_test = X.iloc[test_i]
y_train = y.iloc[train_i]
y_test = y.iloc[test_i]

X_train_const = sm.add_constant(X_train, has_constant="add")
X_test_const = sm.add_constant(X_test, has_constant="add")
X_const = sm.add_constant(X, has_constant="add")

print("\nTRAIN-TEST SPLIT")
print(f"Training rows: {len(train_i)}")
print(f"Testing rows: {len(test_i)}")
print(f"Training matches: {data.iloc[train_i]['MatchID'].nunique()}")
print(f"Testing matches: {data.iloc[test_i]['MatchID'].nunique()}")

# --------------------------------------------------
# 5. MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

ols_train = sm.OLS(y_train, X_train_const).fit()
ols_test_pred = ols_train.predict(X_test_const)

# Full dataset coefficient inference with match-clustered robust SEs
ols_full = sm.OLS(y, X_const).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups},
)

print("\n" + "=" * 60)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 60)
print(ols_full.summary())

# --------------------------------------------------
# 6. POISSON REGRESSION
# --------------------------------------------------

poisson_train = sm.GLM(
    y_train,
    X_train_const,
    family=sm.families.Poisson(),
).fit()

poisson_test_pred = poisson_train.predict(X_test_const)

# Full dataset coefficient inference with match-clustered robust SEs
poisson_full = sm.GLM(
    y,
    X_const,
    family=sm.families.Poisson(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups},
)

print("\n" + "=" * 60)
print("POISSON REGRESSION")
print("=" * 60)
print(poisson_full.summary())

# --------------------------------------------------
# 7. EVALUATE AND COMPARE MODELS ON THE SAME HOLDOUT
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    predicted = np.asarray(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R_squared": r2_score(actual, predicted),
    }

ols_metrics = calculate_metrics(y_test, ols_test_pred)
poisson_metrics = calculate_metrics(y_test, poisson_test_pred)

comparison = pd.DataFrame([
    {
        "Model": "Multiple Linear Regression",
        **ols_metrics,
        "AIC_Full_Data": ols_full.aic,
    },
    {
        "Model": "Poisson Regression",
        **poisson_metrics,
        "AIC_Full_Data": poisson_full.aic,
    },
])

print("\n" + "=" * 60)
print("MODEL COMPARISON — HELD-OUT MATCHES")
print("=" * 60)
print(comparison.round(4).to_string(index=False))
comparison.to_csv(OUT / "model_comparison.csv", index=False)

# --------------------------------------------------
# 8. POISSON DISPERSION CHECK
# --------------------------------------------------

pearson_chi2 = np.sum(poisson_full.resid_pearson ** 2)
dispersion = pearson_chi2 / poisson_full.df_resid

print("\nPOISSON DISPERSION CHECK")
print(f"Pearson chi-square: {pearson_chi2:.4f}")
print(f"Residual degrees of freedom: {poisson_full.df_resid}")
print(f"Dispersion statistic: {dispersion:.4f}")

if dispersion > 1.5:
    print(
        "Possible overdispersion: consider whether a Negative Binomial "
        "model is appropriate, if permitted by the assessment."
    )
elif dispersion < 0.7:
    print("Possible underdispersion: inspect outcome and model fit.")
else:
    print("No strong dispersion warning from this simple diagnostic.")

# --------------------------------------------------
# 9. SAVE COEFFICIENT TABLES
# --------------------------------------------------

def coefficient_table(model, model_name):
    conf = model.conf_int()
    return pd.DataFrame({
        "Model": model_name,
        "Predictor": model.params.index,
        "Coefficient": model.params.values,
        "Robust_SE": model.bse.values,
        "Test_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
        "CI_lower_95": conf[0].values,
        "CI_upper_95": conf[1].values,
    })

all_coef = pd.concat([
    coefficient_table(ols_full, "Multiple Linear Regression"),
    coefficient_table(poisson_full, "Poisson Regression"),
], ignore_index=True)

all_coef.to_csv(OUT / "regression_2_2_coefficients.csv", index=False)

print("\nCOEFFICIENTS")
print(all_coef.round(4).to_string(index=False))

# --------------------------------------------------
# 10. ACTUAL VS PREDICTED PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, ols_test_pred, alpha=0.7, label="Linear Regression")
plt.scatter(
    y_test, poisson_test_pred, alpha=0.7, marker="x",
    label="Poisson Regression",
)

lower = min(
    float(y_test.min()),
    float(np.min(ols_test_pred)),
    float(np.min(poisson_test_pred)),
)
upper = max(
    float(y_test.max()),
    float(np.max(ols_test_pred)),
    float(np.max(poisson_test_pred)),
)
plt.plot([lower, upper], [lower, upper], linestyle="--", label="Ideal prediction")
plt.xlabel("Actual Goals Scored")
plt.ylabel("Predicted Goals Scored")
plt.title("Actual vs Predicted Goals — Regression 2.2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=180)
plt.close()

# --------------------------------------------------
# 11. OLS RESIDUALS VS FITTED
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(ols_full.fittedvalues, ols_full.resid, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted Goals Scored")
plt.ylabel("OLS Residuals")
plt.title("OLS Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(OUT / "residuals_vs_fitted.png", dpi=180)
plt.close()

# --------------------------------------------------
# 12. GOALS DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 6))
max_goals = int(y.max())
plt.hist(
    y,
    bins=np.arange(-0.5, max_goals + 1.5, 1),
    edgecolor="black",
)
plt.xlabel("Goals Scored")
plt.ylabel("Number of Team-Match Observations")
plt.title("Distribution of Goals Scored")
plt.xticks(range(0, max_goals + 1))
plt.tight_layout()
plt.savefig(OUT / "goal_scored_distribution.png", dpi=180)
plt.close()

# --------------------------------------------------
# 13. SAVE SUMMARIES AND HOLDOUT PREDICTIONS
# --------------------------------------------------

summary_text = (
    "REGRESSION 2.2 — MODEL SUMMARIES\n"
    + "=" * 60
    + "\n\nMULTIPLE LINEAR REGRESSION\n"
    + ols_full.summary().as_text()
    + "\n\nPOISSON REGRESSION\n"
    + poisson_full.summary().as_text()
    + "\n\nMODEL COMPARISON ON HELD-OUT MATCHES\n"
    + comparison.round(4).to_string(index=False)
    + f"\n\nPOISSON DISPERSION STATISTIC: {dispersion:.4f}\n"
)

(OUT / "regression_2_2_summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)

predictions = data.iloc[test_i][["MatchID"]].copy()
predictions["Actual_Goals"] = y_test.values
predictions["OLS_Predicted_Goals"] = np.asarray(ols_test_pred)
predictions["Poisson_Predicted_Goals"] = np.asarray(poisson_test_pred)
predictions.to_csv(OUT / "holdout_predictions.csv", index=False)

# --------------------------------------------------
# 14. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)
print(f"Rows analysed: {len(data)}")
print(f"Matches analysed: {data['MatchID'].nunique()}")

print("\nOLS grouped holdout metrics:")
for metric, value in ols_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nPoisson grouped holdout metrics:")
for metric, value in poisson_metrics.items():
    print(f"{metric}: {value:.4f}")

print(f"\nPoisson dispersion: {dispersion:.4f}")
print(f"\nAll output files saved to:\n{OUT}")
print("\nRegression 2.2 script finished successfully.")
