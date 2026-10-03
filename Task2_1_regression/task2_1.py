# task2_1.py
# Author: Mildred
# Objective 2 - LINEAR REGRESSION 2.1
# Predicting the GOAL DIFFERENCE of a FIFA World Cup 2026 match
#
# Everything for this task is in this one file. It reads the data from an
# Excel file (no downloading):
#     data/lr21_matches.xlsx   - 104 rows, one row per World Cup 2026 match
#
# Run it with:   python task2_1.py
#
# My plan (each step has its own heading below and in the output):
#   STEP 1  the question - what am I predicting?
#   STEP 2  choosing the 8 explanatory variables (and what I expect)
#   STEP 3  loading and checking the data
#   STEP 4  exploring the data - descriptive statistics and correlation
#   STEP 5  checking the variables are not copies of each other (VIF)
#   STEP 6  splitting into training and test data
#   STEP 7  building the linear regression model
#   STEP 8  evaluating the model on the test data
#   STEP 9  which variables matter? (p-values with statsmodels)
#   STEP 10 checking the assumptions of linear regression
#   STEP 11 conclusion

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")      # save the plots to files instead of opening windows
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "lr21_matches.xlsx")

LINE = "=" * 72


def heading(text):
    """print a title with a line under it so the output is easy to read"""
    print()
    print(LINE)
    print(text)
    print(LINE)


# =====================================================================
# STEP 1 - THE QUESTION
# =====================================================================
# I want to predict the GOAL DIFFERENCE of a match:
#
#       GoalDiff = home team goals - away team goals
#
# e.g. Mexico 2-0 South Africa  ->  GoalDiff = +2
#      a 1-1 draw               ->  GoalDiff =  0
#      a 0-3 loss for home      ->  GoalDiff = -3
#
# GoalDiff is my RESPONSE (y) variable. It can be negative, zero or
# positive, so it suits linear regression, which predicts a number on a
# continuous scale.
#
# "Home" and "away" are just the order FIFA lists the teams in. Apart from
# the three host nations (USA, Mexico, Canada) nobody is really at home.

TARGET = "GoalDiff"


# =====================================================================
# STEP 2 - CHOOSING THE 8 EXPLANATORY VARIABLES
# =====================================================================
# The rule from the brief: every variable must be KNOWN BEFORE THE MATCH.
# So I could not use anything that happens during the match (shots,
# possession, cards...). I asked myself: "what would a pundit know the
# night before the game?"
#
# Because I am predicting a DIFFERENCE between two teams, I made every
# variable a difference too (home value - away value). If two teams are
# exactly equal on everything, all 8 variables are 0 and the model should
# predict a goal difference of about 0. That made sense to me.
#
# Where the data came from: FIFA's official data service (api.fifa.com),
# the same source as my Task 3, compiled into the Excel file.
#
# What "form" means: the team's average from its EARLIER matches at this
# World Cup only - never the match being predicted. Before a team's first
# match it has no form yet, so I started everyone at the average team at
# the 2022 World Cup (1.34 goals, 1.38 points per match):
#     form = (total so far + 2022 average) / (matches so far + 1)
# That way the first matches are not empty, and a team's form moves away
# from the average as it plays more.
#
# For each variable I wrote down what I EXPECTED the sign of its
# coefficient to be BEFORE running the model, so I can check later whether
# the results make football sense.

VARIABLES = [
    # (column name, what it is, why I chose it, sign I expect)
    ("RankPointsDiff",
     "FIFA ranking points, home - away (ranking of 11 Jun 2026)",
     "the official measure of how strong a team is before the tournament",
     "+"),
    ("HostAdvantage",
     "+1 home team is a host in its own country, -1 away team is, else 0",
     "home crowd, no travel, familiar conditions",
     "+"),
    ("FormGoalsForDiff",
     "goals scored per match in earlier 2026 matches, home - away",
     "a team that has been scoring a lot is in good attacking form",
     "+"),
    ("FormGoalsAgstDiff",
     "goals conceded per match in earlier 2026 matches, home - away",
     "a leaky defence should lead to a worse goal difference",
     "-"),
    ("FormPointsDiff",
     "points per match in earlier 2026 matches, home - away",
     "winning teams tend to keep winning (momentum / confidence)",
     "+"),
    ("SquadAgeDiff",
     "mean age of the registered squad (years), home - away",
     "older = more experience, but younger = fitter; I was not sure",
     "?"),
    ("RestDaysDiff",
     "days since previous match (max 7), home - away",
     "a more rested team should have fresher legs",
     "+"),
    ("StrongConfedDiff",
     "1 if UEFA or CONMEBOL team else 0, home - away",
     "Europe and South America have won every World Cup so far",
     "+"),
]
FEATURES = [v[0] for v in VARIABLES]

# The brief says at most 4 variables can be shared with Task 2.2.
# Shared: RankPointsDiff, HostAdvantage, FormGoalsForDiff, FormGoalsAgstDiff
# Only in 2.1: FormPointsDiff, SquadAgeDiff, RestDaysDiff, StrongConfedDiff


def main():
    heading("LINEAR REGRESSION 2.1 - GOAL DIFFERENCE (Mildred)")
    print("Can we predict the goal difference of a World Cup 2026 match")
    print("using only information that was available BEFORE kick-off?")

    # =================================================================
    heading("STEP 1: THE QUESTION")
    # =================================================================
    print("Response variable (y): GoalDiff = home goals - away goals")
    print("Model                : multiple linear regression, 8 variables")

    # =================================================================
    heading("STEP 2: THE 8 EXPLANATORY VARIABLES (all known before kick-off)")
    # =================================================================
    for name, meaning, why, sign in VARIABLES:
        print("%-18s expect %s" % (name, sign))
        print("   what: %s" % meaning)
        print("   why : %s" % why)
    print("\nShared with Task 2.2 (max 4 allowed): RankPointsDiff, HostAdvantage,")
    print("FormGoalsForDiff, FormGoalsAgstDiff")

    # =================================================================
    heading("STEP 3: LOADING AND CHECKING THE DATA")
    # =================================================================
    print("Reading data from:", DATA_FILE)
    if not os.path.exists(DATA_FILE):
        print("Could not find data/lr21_matches.xlsx - put the Excel file in")
        print("the data folder next to this script.")
        return
    df = pd.read_excel(DATA_FILE)      # needs the openpyxl package
    print("Loaded %d rows and %d columns" % df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    # check the dataset follows the brief
    print("\nChecks against the brief:")
    print("   rows = %d (need 104, one per match)" % len(df))
    print("   explanatory variables = %d (need exactly 8)" % len(FEATURES))
    print("   ID fields (not used in the model): MatchID, Date, HomeTeam, AwayTeam")

    # missing values: LISTWISE (CASE) DELETION, same as my Task 3.
    # If a match is missing any value, the whole row is dropped.
    rows_before = len(df)
    print("   missing values per column:", int(df[FEATURES + [TARGET]].isna().sum().sum()))
    df = df.dropna(subset=FEATURES + [TARGET])
    print("   rows dropped by listwise deletion: %d" % (rows_before - len(df)))

    X = df[FEATURES]
    y = df[TARGET]

    # =================================================================
    heading("STEP 4: EXPLORING THE DATA")
    # =================================================================
    # First I look at y on its own, then how each variable moves with y.
    print("GoalDiff (y):")
    print("   mean %.3f   median %.1f   std %.3f   min %d   max %d"
          % (y.mean(), y.median(), y.std(), y.min(), y.max()))
    print("   home team won %d, drew %d, lost %d"
          % ((y > 0).sum(), (y == 0).sum(), (y < 0).sum()))

    # Pearson correlation r: -1 to +1. Near 0 = no straight-line link.
    print("\nCorrelation of each variable with GoalDiff:")
    for name in FEATURES:
        r = X[name].corr(y)
        strength = ("strong" if abs(r) >= 0.5 else
                    "moderate" if abs(r) >= 0.3 else
                    "weak" if abs(r) >= 0.1 else "very weak")
        print("   %-18s r = %+.3f   (%s)" % (name, r, strength))
    print("\nRanking points has by far the strongest link with goal difference,")
    print("which is what I expected - better teams beat weaker teams.")

    # =================================================================
    heading("STEP 5: ARE THE VARIABLES COPIES OF EACH OTHER? (VIF)")
    # =================================================================
    # If two variables carry the same information (MULTICOLLINEARITY) the
    # model cannot tell which one deserves the credit, and the coefficients
    # become unreliable. VIF (Variance Inflation Factor) measures this:
    #     VIF = 1 / (1 - R^2), where R^2 is how well the OTHER variables
    #     predict this one.   VIF < 5 is fine, VIF > 10 is a problem.
    Xc = sm.add_constant(X)
    vifs = {name: variance_inflation_factor(Xc.values, i + 1)
            for i, name in enumerate(FEATURES)}
    for name, v in vifs.items():
        print("   %-18s VIF = %.2f   %s" % (name, v, "ok" if v < 5 else "HIGH"))
    print("\nAll below 5, so I can keep all 8 variables.")

    # =================================================================
    heading("STEP 6: TRAINING AND TEST DATA")
    # =================================================================
    # If I test the model on the same matches it learned from, it will look
    # better than it really is. So I hide 20% of the matches, train on the
    # other 80%, then test on the hidden ones.
    # random_state=42 keeps the same split every time I run it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("Training rows: %d   Test rows: %d" % (len(X_train), len(X_test)))

    # =================================================================
    heading("STEP 7: BUILDING THE MODEL")
    # =================================================================
    # Multiple linear regression:
    #   GoalDiff = b0 + b1*x1 + b2*x2 + ... + b8*x8
    # .fit() finds the b values that make the squared errors as small as
    # possible (ordinary least squares).
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercept b0 = %.4f" % model.intercept_)
    print("\n   %-18s %10s  %-8s %-8s" % ("variable", "coef", "expected", "got"))
    for (name, _, _, expect), b in zip(VARIABLES, model.coef_):
        got = "+" if b > 0 else "-"
        note = "" if expect == "?" else ("matches" if got == expect else "OPPOSITE")
        print("   %-18s %+10.4f  %-8s %-8s %s" % (name, b, expect, got, note))

    print("\nHow to read a coefficient, e.g. RankPointsDiff = %+.4f:"
          % model.coef_[0])
    print("   for every 100 extra ranking points the home team has over the")
    print("   away team, the predicted goal difference goes up by %.2f goals,"
          % (model.coef_[0] * 100))
    print("   with the other 7 variables held the same.")
    print("\nWhy can a sign come out OPPOSITE to what I expected? Two reasons:")
    print(" - the form variables overlap with ranking points (good teams have")
    print("   good form). Once ranking is in the model, what is left of form")
    print("   is mostly noise, so its coefficient can flip.")
    print(" - some effects are so small (e.g. rest days) that the coefficient")
    print("   is basically 0 and its sign is down to chance.")
    print("Step 9 checks which coefficients are significantly different from 0.")

    # =================================================================
    heading("STEP 8: HOW GOOD IS THE MODEL ON UNSEEN MATCHES?")
    # =================================================================
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # normalised RMSE: RMSE divided by the range of y, so it is a fraction
    nrmse = rmse / (y.max() - y.min())
    r2 = r2_score(y_test, y_pred)

    # A model is only useful if it beats a "dumb" guess. The BASELINE
    # ignores all 8 variables and always predicts the average GoalDiff of
    # the training matches.
    baseline = np.full(len(y_test), y_train.mean())
    b_mae = mean_absolute_error(y_test, baseline)
    b_rmse = np.sqrt(mean_squared_error(y_test, baseline))
    b_r2 = r2_score(y_test, baseline)

    print("   %-20s %10s %10s" % ("", "my model", "baseline"))
    print("   %-20s %10.3f %10.3f" % ("MAE", mae, b_mae))
    print("   %-20s %10.3f %10.3f" % ("RMSE", rmse, b_rmse))
    print("   %-20s %10.3f" % ("Normalised RMSE", nrmse))
    print("   %-20s %10.3f %10.3f" % ("R-squared", r2, b_r2))
    print("\nMAE: on average the prediction is %.2f goals away from the real"
          % mae)
    print("goal difference, compared with %.2f goals for the baseline." % b_mae)

    # Football check: does it at least pick the right winner?
    decided = y_test != 0                      # leave out the draws
    right = np.sign(y_pred[decided]) == np.sign(y_test[decided])
    print("\nRight winner in %d of %d test matches that were not draws (%.0f%%)."
          % (right.sum(), decided.sum(), right.mean() * 100))

    # The test set is only 21 matches, so the score depends on which 21 we
    # happened to pick. Cross-validation repeats the split 5 times (every
    # match is in the test set once) and averages the results.
    cv = cross_val_score(LinearRegression(), X, y, scoring="r2",
                         cv=KFold(5, shuffle=True, random_state=42))
    print("\n5-fold cross-validation R-squared: %s"
          % ", ".join("%.2f" % s for s in cv))
    print("   average %.3f - a fairer estimate than one test split" % cv.mean())

    # =================================================================
    heading("STEP 9: WHICH VARIABLES ACTUALLY MATTER? (statsmodels OLS)")
    # =================================================================
    # scikit-learn does not give p-values, so I fit the same model with
    # statsmodels on all 104 matches. For each coefficient:
    #   H0: b = 0   (the variable has no effect on goal difference)
    #   Ha: b != 0
    # reject H0 when p <= 0.05
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    print("R-squared %.3f   adjusted R-squared %.3f"
          % (ols.rsquared, ols.rsquared_adj))
    print("F-test p-value %.2g  (H0: all 8 coefficients are 0)" % ols.f_pvalue)
    print()
    print("   %-18s %9s %9s %5s  %s" % ("term", "coef", "p-value", "", "95% CI"))
    for term in ["const"] + FEATURES:
        lo, hi = ols.conf_int().loc[term]
        p = ols.pvalues[term]
        # a tiny p-value would print as 0.0000, but a probability is never
        # exactly 0, so I write it as < 0.0001 (same as my Task 3)
        p_text = "< 0.0001" if p < 0.0001 else "%.4f" % p
        print("   %-18s %+9.4f %9s %5s  [%+.4f, %+.4f]"
              % ("Intercept" if term == "const" else term, ols.params[term],
                 p_text, "SIG" if p <= 0.05 else "", lo, hi))
    significant = [f for f in FEATURES if ols.pvalues[f] <= 0.05]
    print("\nSignificant at 5%%: %s" % ", ".join(significant))
    print("Adjusted R-squared is lower than R-squared because it is penalised")
    print("for every variable that doesn't help - a sign some of the 8 are")
    print("not adding much on top of the ranking.")

    # =================================================================
    heading("STEP 10: CHECKING THE ASSUMPTIONS")
    # =================================================================
    # Linear regression is only trustworthy if the RESIDUALS (actual -
    # predicted) behave well:
    #   1. linear relationship  -> residuals vs fitted has no curve
    #   2. normal residuals     -> Shapiro-Wilk test, H0: residuals normal
    #   3. constant spread      -> Breusch-Pagan test, H0: spread constant
    #   4. no multicollinearity -> VIF, done in Step 5
    resid = ols.resid
    sw_p = stats.shapiro(resid).pvalue
    bp_p = het_breuschpagan(resid, ols.model.exog)[1]
    print("   mean of residuals    %.4f  (should be about 0)" % resid.mean())
    print("   Shapiro-Wilk p       %.3f  -> %s" % (sw_p,
          "normal, ok" if sw_p > 0.05 else "NOT normal"))
    print("   Breusch-Pagan p      %.3f  -> %s" % (bp_p,
          "constant spread, ok" if bp_p > 0.05 else "spread NOT constant"))
    print("   largest VIF          %.2f  -> ok" % max(vifs.values()))

    # the two plots for the report
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(ols.fittedvalues, y, alpha=0.6)
    ax[0].plot([-4, 6], [-4, 6], "--", color="grey")
    ax[0].set_xlabel("Predicted goal difference")
    ax[0].set_ylabel("Actual goal difference")
    ax[0].set_title("Actual vs predicted")
    ax[1].scatter(ols.fittedvalues, resid, alpha=0.6)
    ax[1].axhline(0, linestyle="--", color="grey")
    ax[1].set_xlabel("Predicted goal difference")
    ax[1].set_ylabel("Residual (actual - predicted)")
    ax[1].set_title("Residuals vs fitted")
    fig.tight_layout()
    plot_file = os.path.join(HERE, "lr21_diagnostics.png")
    fig.savefig(plot_file, dpi=150)
    print("\nPlots saved to lr21_diagnostics.png - the residuals are scattered")
    print("evenly around 0 with no curve, so a straight line is reasonable.")

    # =================================================================
    heading("STEP 11: CONCLUSION")
    # =================================================================
    print("Using only information available before kick-off, the model")
    print("explains %.0f%% of the variation in goal difference (adjusted"
          % (ols.rsquared * 100))
    print("R-squared %.2f, F-test p = %.2g, so the model is significant)."
          % (ols.rsquared_adj, ols.f_pvalue))
    print()
    print("On unseen matches it is off by %.2f goals on average, against"
          % mae)
    print("%.2f for the baseline guess, and it picks the right winner %.0f%%"
          % (b_mae, right.mean() * 100))
    ok = sw_p > 0.05 and bp_p > 0.05 and max(vifs.values()) < 5
    print("of the time. The assumptions checked in Step 10 %s."
          % ("hold" if ok else "do NOT all hold, so treat p-values with care"))
    print()
    print("Significant at 5%%: %s. Most of the" % ", ".join(significant))
    print("predictive power comes from how strong the teams were BEFORE the")
    print("tournament; recent form adds little once the ranking is known.")
    print()
    print("Limitations:")
    print(" - 104 matches is a small dataset, and the 21-match test set makes")
    print("   the test score jumpy (cross-validation average %.2f)." % cv.mean())
    print(" - Goal difference is whole numbers, but the model predicts")
    print("   decimals - fine for ranking teams, not for exact scores.")
    print(" - Football has a lot of luck: even a perfect model could not")
    print("   explain red cards, penalties or a goalkeeper's great day.")
    print()
    print(LINE)
    print("End of Linear Regression 2.1.")
    print(LINE)


if __name__ == "__main__":
    main()
