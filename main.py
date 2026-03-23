from xgboost import XGBClassifier
from dataset import create_dataset
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


train_transactions = pd.read_csv("train_transactions.csv")
train_users = pd.read_csv("train_users.csv")

full_dataset = create_dataset(train_transactions, train_users)

feature_cols = [c for c in full_dataset.columns if c not in ["id_user", "is_fraud"]]

X = full_dataset[feature_cols].copy()
y = full_dataset["is_fraud"].copy()

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)

X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)


train_feature_cols = X.columns.tolist()
print(train_feature_cols)


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_t = 0.5
    best_f1 = -1

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred)
        if score > best_f1:
            best_f1 = score
            best_t = t

    return best_t, best_f1


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X), dtype=float)

feature_importance_folds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=900,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.82,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=2.0,
        reg_alpha=0.2,
        scale_pos_weight=scale_pos_weight,
        random_state=42 + fold,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    oof_proba[va_idx] = model.predict_proba(X_va)[:, 1]

    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    fold_importance = pd.DataFrame({
        "feature": list(score_dict.keys()),
        "gain_importance": list(score_dict.values())
    })

    feature_importance_folds.append(fold_importance)

if feature_importance_folds:
    importance_df = pd.concat(feature_importance_folds, ignore_index=True)

    importance_df = (
        importance_df
        .groupby("feature", as_index=False)["gain_importance"]
        .mean()
        .sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 100 features by average gain importance:")
    print(importance_df.head(100))

    importance_df.to_csv("xgb_feature_importance.csv", index=False)

best_t, best_f1 = find_best_threshold(y, oof_proba)

print("Best threshold:", best_t)
print("Best OOF F1:", best_f1)

full_dataset["xgb_oof_proba"] = oof_proba
full_dataset["xgb_pred"] = (full_dataset["xgb_oof_proba"] >= best_t).astype(int)

missed_frauds = full_dataset[
    (full_dataset["is_fraud"] == 1) &
    (full_dataset["xgb_pred"] == 0)
].copy()

missed_frauds = missed_frauds.sort_values("xgb_oof_proba", ascending=False).reset_index(drop=True)
print("Missed frauds:", len(missed_frauds))

catched_frauds = full_dataset[
    (full_dataset["is_fraud"] == 1) &
    (full_dataset["xgb_pred"] == 1)
].copy()

catched_frauds = catched_frauds.sort_values("xgb_oof_proba", ascending=False).reset_index(drop=True)
print("Catched frauds:", len(catched_frauds))

miss_catched_frauds = full_dataset[
    (full_dataset["is_fraud"] == 0) &
    (full_dataset["xgb_pred"] == 1)
].copy()

miss_catched_frauds = miss_catched_frauds.sort_values("xgb_oof_proba", ascending=False).reset_index(drop=True)
print("Miss catched frauds:", len(miss_catched_frauds))

# Save suspicious legit users for manual inspection
suspicious_legit_ids = miss_catched_frauds["id_user"].unique()

suspicious_legit_users = train_users[train_users["id_user"].isin(suspicious_legit_ids)].copy()
suspicious_legit_tx = train_transactions[train_transactions["id_user"].isin(suspicious_legit_ids)].copy()

suspicious_legit_tx["timestamp_tr"] = pd.to_datetime(suspicious_legit_tx["timestamp_tr"], errors="coerce", utc=True)
suspicious_legit_users["timestamp_reg"] = pd.to_datetime(suspicious_legit_users["timestamp_reg"], errors="coerce", utc=True)

suspicious_legit_tx = suspicious_legit_tx.merge(
    suspicious_legit_users,
    on="id_user",
    how="left",
    suffixes=("_tx", "_user")
)

suspicious_legit_tx = suspicious_legit_tx.merge(
    miss_catched_frauds[["id_user", "xgb_oof_proba", "xgb_pred"]],
    on="id_user",
    how="left"
)

suspicious_legit_tx = suspicious_legit_tx.sort_values(
    ["xgb_oof_proba", "id_user", "timestamp_tr"],
    ascending=[False, True, True]
).reset_index(drop=True)

suspicious_legit_tx["tx_order"] = suspicious_legit_tx.groupby("id_user").cumcount() + 1
suspicious_legit_tx["prev_timestamp_tr"] = suspicious_legit_tx.groupby("id_user")["timestamp_tr"].shift(1)
suspicious_legit_tx["minutes_since_prev_tx"] = (
    (suspicious_legit_tx["timestamp_tr"] - suspicious_legit_tx["prev_timestamp_tr"]).dt.total_seconds() / 60.0
)

suspicious_legit_tx.to_csv("suspicious_legit_users.csv", index=False)


# -------------------------
# false positives
# -------------------------
fp_users = full_dataset[
    (
        (full_dataset["is_fraud"] == 0) &
        (full_dataset["xgb_pred"] == 1)
    ) |
    (
        (full_dataset["is_fraud"] == 1) &
        (full_dataset["xgb_pred"] == 0)
    )
][["id_user", "xgb_oof_proba", "xgb_pred"]].drop_duplicates("id_user")

fp_tx = full_dataset[full_dataset["id_user"].isin(fp_users["id_user"])].copy()

fp_tx = fp_tx.merge(
    fp_users,
    on="id_user",
    how="left"
)

fp_tx = fp_tx.sort_values(
    ["id_user"]
).reset_index(drop=True)
fp_tx.drop(columns=["is_fraud", "xgb_oof_proba_x", "xgb_pred_x", "xgb_oof_proba_y", "xgb_pred_y"])

model.save_model("xgb_model.json")
