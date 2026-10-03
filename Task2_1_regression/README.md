# Linear Regression 2.1 — Goal difference (Objective 2)

Predicts the **goal difference** (home goals − away goals) of a FIFA World Cup
2026 match using only information available **before kick-off**.

Everything is in one file, `task2_1.py`. It reads `data/lr21_matches.xlsx`
and prints each step with an explanation.

## How to run
```
pip install -r requirements.txt
python task2_1.py
```

## The dataset — `data/lr21_matches.xlsx`
104 rows (one per match), compiled from FIFA's official data service
(api.fifa.com), the same source as Task 3.

| Column | Meaning |
|---|---|
| MatchID, Date, HomeTeam, AwayTeam | ID fields, not used in the model |
| **RankPointsDiff** ★ | FIFA ranking points (ranking of 11 Jun 2026), home − away |
| **HostAdvantage** ★ | +1 home team is a host in its own country, −1 away team is, else 0 |
| **FormGoalsForDiff** ★ | goals scored per match in earlier 2026 matches, home − away |
| **FormGoalsAgstDiff** ★ | goals conceded per match in earlier 2026 matches, home − away |
| **FormPointsDiff** | points per match in earlier 2026 matches, home − away |
| **SquadAgeDiff** | mean age of the registered squad, home − away |
| **RestDaysDiff** | days since previous match (max 7), home − away |
| **StrongConfedDiff** | UEFA/CONMEBOL team (1/0), home − away |
| GoalDiff | **response variable**: home goals − away goals |

★ = shared with Task 2.2 (the maximum of 4 allowed).

"Form" only uses a team's **earlier** matches. Before a team's first match it
starts at the 2022 World Cup average, so the first matches are not empty.

## Steps in the code
1. The question · 2. Choosing the 8 variables (with the sign I expected) ·
3. Loading and checking the data · 4. Descriptive statistics and correlation ·
5. Multicollinearity (VIF) · 6. 80/20 train/test split · 7. Fitting the model ·
8. Evaluating it (MAE, RMSE, normalised RMSE, R², against a baseline, plus 5-fold CV) ·
9. p-values (statsmodels OLS) · 10. Assumption checks + plot · 11. Conclusion

## Result
```
R² 0.458, adjusted R² 0.412, F-test p = 4.9e-10
Test set: MAE 1.36 goals (baseline 1.84), R² 0.40, right winner 81%
5-fold CV R² 0.30
Significant: RankPointsDiff (p < 0.0001)
Assumptions: residuals normal (p = 0.46), constant spread (p = 0.09), max VIF 3.2
```
The pre-tournament FIFA ranking carries most of the predictive power, and
tournament form adds little once the ranking is known.
