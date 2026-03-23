# dataset.py

import re
import numpy as np
import pandas as pd


def create_dataset(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:

    # -------------------------
    # basic cleanup
    # -------------------------
    transactions["timestamp_tr"] = pd.to_datetime(transactions["timestamp_tr"], errors="coerce", utc=True)
    users["timestamp_reg"] = pd.to_datetime(users["timestamp_reg"], errors="coerce", utc=True)

    transactions["amount"] = pd.to_numeric(transactions.get("amount", 0), errors="coerce").fillna(0.0)

    transactions_cols = [
        "status",
        "error_group",
        "card_mask_hash",
        "card_holder",
        "transaction_type",
        "card_country",
        "payment_country",
    ]
    for col in transactions_cols:
        if col not in transactions.columns:
            transactions[col] = "missing"
        transactions[col] = transactions[col].fillna("missing").astype(str).str.strip().str.lower()

    users_cols = ["traffic_type", "gender", "reg_country", "email"]
    for col in users_cols:
        if col not in users.columns:
            users[col] = "missing"
        users[col] = users[col].fillna("missing").astype(str).str.strip().str.lower()

    # -------------------------
    # simple normalized fields
    # -------------------------
    transactions["card_holder_norm"] = transactions["card_holder"].astype(str).str.lower()
    transactions["card_holder_norm"] = transactions["card_holder_norm"].str.replace(r"[^a-z0-9\s]", " ", regex=True)
    transactions["card_holder_norm"] = transactions["card_holder_norm"].str.replace(r"\b(mr|mrs|ms|miss|dr)\b", " ", regex=True)
    transactions["card_holder_norm"] = transactions["card_holder_norm"].str.replace(r"\b[a-z]\b", " ", regex=True)
    transactions["card_holder_norm"] = transactions["card_holder_norm"].str.replace(r"\s+", " ", regex=True).str.strip()
    transactions["card_holder_norm"] = transactions["card_holder_norm"].replace("", "missing").fillna("missing")

    transactions["error_group"] = transactions["error_group"]
    transactions.loc[transactions["error_group"].isin(["", "nan", "none", "missing"]), "error_group"] = "none"
    transactions.loc[transactions["error_group"].str.contains("fraud|antifraud", na=False), "error_group"] = "fraud"
    transactions.loc[transactions["error_group"].str.contains("3ds", na=False), "error_group"] = "3ds"
    transactions.loc[transactions["error_group"].str.contains("insufficient", na=False), "error_group"] = "insuff"
    transactions.loc[transactions["error_group"].str.contains("do not honor", na=False), "error_group"] = "dnh"
    transactions.loc[transactions["error_group"].str.contains("limit exceeded", na=False), "error_group"] = "limit"
    transactions.loc[transactions["error_group"].str.contains("issuer decline", na=False), "error_group"] = "issuer_decline"
    transactions.loc[transactions["error_group"].str.contains("cvv", na=False), "error_group"] = "cvv"
    transactions.loc[transactions["error_group"].str.contains("card problem", na=False), "error_group"] = "card_problem"
    transactions.loc[transactions["error_group"].str.contains("invalid", na=False), "error_group"] = "invalid"
    transactions.loc[transactions["error_group"].str.contains("technical", na=False), "error_group"] = "technical"

    known_err = {
        "none", "fraud", "3ds", "insuff", "dnh", "limit",
        "issuer_decline", "cvv", "card_problem", "invalid", "technical"
    }
    transactions.loc[~transactions["error_group"].isin(known_err), "error_group"] = "other"

    # -------------------------
    # merge users info
    # -------------------------s
    keep_user_cols = ["id_user", "timestamp_reg", "traffic_type", "gender", "reg_country", "email"]
    keep_user_cols = [c for c in keep_user_cols if c in users.columns]

    transactions = transactions.merge(users[keep_user_cols], on="id_user", how="left")

    transactions = transactions.sort_values(["id_user", "timestamp_tr"], kind="mergesort").reset_index(drop=True)

    g = transactions.groupby("id_user", sort=False)

    # -------------------------
    # transaction-level features
    # -------------------------
    transactions["is_success"] = transactions["status"].eq("success").astype("int8")
    transactions["is_fail"] = transactions["status"].eq("fail").astype("int8")
    transactions["is_fraud_error"] = transactions["error_group"].eq("fraud").astype("int8")

    transactions["seconds_from_registration"] = (transactions["timestamp_tr"] - transactions["timestamp_reg"]).dt.total_seconds()
    transactions["from_last_transaction_seconds"] = g["timestamp_tr"].diff().dt.total_seconds()
    transactions["transactions_order"] = g.cumcount() + 1

    transactions["reg_vs_card_country_match"] = transactions["reg_country"].eq(transactions["card_country"]).astype("int8")
    transactions["reg_vs_payment_country_match"] = transactions["reg_country"].eq(transactions["payment_country"]).astype("int8")
    transactions["card_vs_payment_country_match"] = transactions["card_country"].eq(transactions["payment_country"]).astype("int8")

    # -------------------------
    # main user features
    # -------------------------
    out = (
        transactions.groupby("id_user", sort=False)
        .agg(
            transaction_number=("id_user", "size"),
            success_count=("is_success", "sum"),
            fail_count=("is_fail", "sum"),
            antifraud_count=("is_fraud_error", "sum"),

            amount_mean=("amount", "mean"),
            amount_std=("amount", "std"),
            amount_min=("amount", "min"),
            amount_max=("amount", "max"),

            unique_card_mask_hash_count=("card_mask_hash", "nunique"),
            unique_card_holder_norm_count=("card_holder_norm", "nunique"),
            unique_error_group_count=("error_group", "nunique"),

            reg_vs_card_country_match_rate=("reg_vs_card_country_match", "mean"),
            reg_vs_payment_country_match_rate=("reg_vs_payment_country_match", "mean"),
            card_vs_payment_country_match_rate=("card_vs_payment_country_match", "mean"),

            first_transactions_after_reg_sec=("seconds_from_registration", "min"),
            last_transactions_after_reg_sec=("seconds_from_registration", "max"),
            from_last_transaction_mean=("from_last_transaction_seconds", "mean"),
        )
        .reset_index()
    )

    transactionsn = out["transaction_number"].replace(0, np.nan)
    out["fail_rate"] = out["fail_count"] / transactionsn
    out["success_rate"] = out["success_count"] / transactionsn
    out["fraud_error_rate"] = out["antifraud_count"] / transactionsn

    active_span_sec = (out["last_transactions_after_reg_sec"] - out["first_transactions_after_reg_sec"]).fillna(0).clip(lower=1)
    out["fails_per_hour_active"] = out["fail_count"] / (active_span_sec / 3600.0)

    out["country_contradiction_score"] = (
        (1 - out["reg_vs_card_country_match_rate"]) +
        (1 - out["reg_vs_payment_country_match_rate"]) +
        (1 - out["card_vs_payment_country_match_rate"])
    ) / 3

    out["fail_before_success_pressure"] = np.where(
        out["success_count"] != 0,
        out["fail_count"] / out["success_count"],
        0
    )
    out["card_pressure_contradiction"] = np.where(
        out["success_count"] != 0,
        out["unique_card_mask_hash_count"] / out["success_count"],
        0
    )

    out["geo_card_fail_interaction"] = out["country_contradiction_score"] * out["fail_before_success_pressure"]
    out["geo_holder_interaction"] = out["country_contradiction_score"] * out["unique_card_holder_norm_count"]

    out["holder_per_card_ratio"] = np.where(
        out["unique_card_mask_hash_count"] != 0,
        out["unique_card_holder_norm_count"] / out["unique_card_mask_hash_count"],
        0
    )
    out["card_per_holder_ratio"] = np.where(
        out["unique_card_holder_norm_count"] != 0,
        out["unique_card_mask_hash_count"] / out["unique_card_holder_norm_count"],
        0
    )

    # -------------------------
    # count changes
    # -------------------------
    changes = (
        transactions.groupby("id_user", sort=False)
        .agg(
            holder_norm_change_count=("card_holder_norm", lambda s: (s.fillna("missing").astype(str) != s.fillna("missing").astype(str).shift(1)).sum() - 1 if len(s) > 0 else 0),
            transaction_type_change_count=("transaction_type", lambda s: (s.fillna("missing").astype(str) != s.fillna("missing").astype(str).shift(1)).sum() - 1 if len(s) > 0 else 0),
            payment_country_change_count=("payment_country", lambda s: (s.fillna("missing").astype(str) != s.fillna("missing").astype(str).shift(1)).sum() - 1 if len(s) > 0 else 0),
        )
        .reset_index()
    )
    out = out.merge(changes, on="id_user", how="left")
    out["transaction_type_fail_interaction"] = out["transaction_type_change_count"] * out["fail_before_success_pressure"]

    # -------------------------
    # top categories
    # -------------------------
    for src_col, out_col in [
        ("error_group", "top_error_group"),
        ("transaction_type", "top_transaction_type"),
        ("payment_country", "top_payment_country"),
        ("card_country", "top_card_country"),
    ]:
        top_df = (
            transactions.loc[transactions[src_col].notna(), ["id_user", src_col]]
            .groupby(["id_user", src_col], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
            .sort_values(["id_user", "cnt", src_col], ascending=[True, False, True], kind="mergesort")
            .drop_duplicates("id_user")[["id_user", src_col]]
            .rename(columns={src_col: out_col})
        )
        out = out.merge(top_df, on="id_user", how="left")

    # -------------------------
    # first session features
    # -------------------------
    first_part = transactions[(transactions["transactions_order"] <= 5) | (transactions["seconds_from_registration"] <= 3600)].copy()
    fs = (
        first_part.groupby("id_user", sort=False)
        .agg(
            fs_transactions_count=("id_user", "size"),
            fs_fail_count=("is_fail", "sum"),
            fs_fraud_error_count=("is_fraud_error", "sum"),
        )
        .reset_index()
    )
    fs_denom = fs["fs_transactions_count"].replace(0, np.nan)
    fs["fs_fail_rate"] = fs["fs_fail_count"] / fs_denom
    fs["fs_fraud_error_rate"] = fs["fs_fraud_error_count"] / fs_denom
    out = out.merge(fs, on="id_user", how="left")

    # -------------------------
    # last10 features
    # -------------------------
    last10_part = transactions.groupby("id_user", sort=False, group_keys=False).tail(10).copy()
    last10 = (
        last10_part.groupby("id_user", sort=False)
        .agg(
            last10_transactions_count=("id_user", "size"),
            last10_fail_count=("is_fail", "sum"),
            last10_success_count=("is_success", "sum"),
            last10_fraud_error_count=("is_fraud_error", "sum"),
            last10_unique_cards=("card_mask_hash", "nunique"),
            last10_unique_holders=("card_holder_norm", "nunique"),
            last10_unique_transactions_types=("transaction_type", "nunique"),
        )
        .reset_index()
    )
    last10_denom = last10["last10_transactions_count"].replace(0, np.nan)
    last10["last10_fail_rate"] = last10["last10_fail_count"] / last10_denom
    last10["last10_fraud_error_rate"] = last10["last10_fraud_error_count"] / last10_denom
    out = out.merge(last10, on="id_user", how="left")

    # -------------------------
    # last20 features
    # -------------------------
    last20_part = transactions.groupby("id_user", sort=False, group_keys=False).tail(20).copy()
    last20 = (
        last20_part.groupby("id_user", sort=False)
        .agg(
            last20_transactions_count=("id_user", "size"),
            last20_fail_count=("is_fail", "sum"),
            last20_success_count=("is_success", "sum"),
            last20_fraud_error_count=("is_fraud_error", "sum"),
            last20_unique_cards=("card_mask_hash", "nunique"),
            last20_unique_holders=("card_holder_norm", "nunique"),
            last20_unique_transactions_types=("transaction_type", "nunique"),
        )
        .reset_index()
    )
    last20_denom = last20["last20_transactions_count"].replace(0, np.nan)
    last20["last20_fail_rate"] = last20["last20_fail_count"] / last20_denom
    last20["last20_fraud_error_rate"] = last20["last20_fraud_error_count"] / last20_denom
    out = out.merge(last20, on="id_user", how="left")

    out["last10_burst_pressure"] = (
        out["last10_fail_count"] +
        2.0 * out["last10_fraud_error_count"] +
        out["last10_unique_cards"]
    )

    out["last20_burst_pressure"] = (
        out["last20_fail_count"] +
        2.0 * out["last20_fraud_error_count"] +
        out["last20_unique_cards"]
    )

    # -------------------------
    # rolling window features
    # -------------------------
    rolling_rows = []
    for user_id, g_user in transactions.groupby("id_user", sort=False):
        g_user = g_user.sort_values("timestamp_tr").copy()
        g_user = g_user[g_user["timestamp_tr"].notna()].copy()

        if g_user.empty:
            rolling_rows.append({
                "id_user": user_id,
                "max_fails_in_1h": 0,
                "max_unique_cards_in_1h": 0,
                "max_fails_in_6h": 0,
                "max_unique_cards_in_6h": 0,
                "max_fails_in_24h": 0,
                "max_unique_cards_in_24h": 0,
            })
            continue

        times = (g_user["timestamp_tr"].astype("int64") // 10**9).to_numpy()
        fails = g_user["is_fail"].astype(int).to_numpy()
        cards = g_user["card_mask_hash"].fillna("missing").astype(str).tolist()

        result = {"id_user": user_id}

        for window_sec, fail_name, card_name in [
            (3600, "max_fails_in_1h", "max_unique_cards_in_1h"),
            (6 * 3600, "max_fails_in_6h", "max_unique_cards_in_6h"),
            (24 * 3600, "max_fails_in_24h", "max_unique_cards_in_24h"),
        ]:
            left = 0
            best_fails = 0
            best_cards = 0

            for right in range(len(g_user)):
                while times[right] - times[left] > window_sec:
                    left += 1

                fail_cnt = int(fails[left:right + 1].sum())
                uniq_cards = len(set(cards[left:right + 1]))

                if fail_cnt > best_fails:
                    best_fails = fail_cnt
                if uniq_cards > best_cards:
                    best_cards = uniq_cards

            result[fail_name] = best_fails
            result[card_name] = best_cards

        rolling_rows.append(result)

    rolling_df = pd.DataFrame(rolling_rows)
    out = out.merge(rolling_df, on="id_user", how="left")

    # -------------------------
    # merge userser categorical info
    # -------------------------
    users_simple = users[["id_user", "traffic_type", "gender", "reg_country", "email"]].copy()
    out = out.merge(users_simple, on="id_user", how="left")

    out["email_domain"] = (
        out["email"].fillna("missing").astype(str).str.split("@").str[-1].replace("", "missing")
    )
    
    out["is_orgainc"] = (
        out["traffic_type"] == "organic"
    )

    # -------------------------
    # cleanup
    # -------------------------
    cat_cols = [
        "traffic_type",
        "gender",
        "reg_country",
        "email_domain",
        "top_error_group",
        "top_transaction_type",
        "top_payment_country",
        "top_card_country",
    ]
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("missing").astype(str)

    out = out.drop(columns=["email"], errors="ignore")

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-1)

    int_cols = out.select_dtypes(include=["int64", "int32", "int16"]).columns
    float_cols = out.select_dtypes(include=["float64", "float32"]).columns

    for col in int_cols:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], downcast="float")

    return out