"""Contract-level freight-rate forecasting using a generic public schema."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from feature_engineering import add_calendar_features, add_lag_rolling_features

CATEGORIES = ["origin_dc", "lane_group", "customer_type", "vehicle_type"]
ENCODE_COLUMNS = ["vehicle_type", "customer_type"]
FEATURES = CATEGORIES + [
    "distance_km", "weight_ton", "is_weekend", "is_holiday",
    "is_near_holiday", "is_month_end", "is_quarter_end",
    "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30",
] + [f"{c}_te" for c in ENCODE_COLUMNS]
REQUIRED = [
    "contract_id", "order_date", "origin_dc", "dest_province", "dest_city",
    "customer_type", "vehicle_type", "distance_km", "weight_ton",
]

def validate_shipments(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    needed = REQUIRED + (["freight"] if require_target else [])
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame.copy()
    if df.empty:
        raise ValueError("Input must contain at least one shipment.")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="raise").dt.normalize()
    for col in ["contract_id", "order_date", "origin_dc", "dest_province", "dest_city", "customer_type"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} contains missing or empty values.")
    for col in ["weight_ton", "distance_km"] + (["freight"] if require_target else []):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must contain finite numbers.")
    if (df["weight_ton"] <= 0).any() or (df["distance_km"] < 0).any():
        raise ValueError("Weights must be positive and distances nonnegative.")
    if require_target and (df["freight"] < 0).any():
        raise ValueError("Freight charges must be nonnegative.")
    for col in ["order_date", "origin_dc", "vehicle_type"]:
        if (df.groupby("contract_id")[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"{col} must be consistent within each contract.")
    return df.reset_index(drop=True)

def aggregate_contracts(lines: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    df = validate_shipments(lines, require_target)
    rules = {
        "order_date": ("order_date", "first"),
        "origin_dc": ("origin_dc", "first"),
        "customer_type": ("customer_type", lambda x: x.mode().iloc[0]),
        "vehicle_type": ("vehicle_type", "first"),
        "weight_ton": ("weight_ton", "sum"),
        "distance_km": ("distance_km", "max"),
    }
    if require_target:
        rules["freight"] = ("freight", "sum")
    out = df.groupby("contract_id").agg(**rules).reset_index()
    farthest = df.sort_values("distance_km", ascending=False, kind="stable").drop_duplicates("contract_id")
    out = out.merge(farthest[["contract_id", "dest_province", "dest_city"]],
                    on="contract_id", validate="one_to_one")
    out["unit_price"] = out["freight"] / out["weight_ton"] if require_target else np.nan
    return out.sort_values(["order_date", "contract_id"]).reset_index(drop=True)

def _features(contracts: pd.DataFrame, holidays=None) -> pd.DataFrame:
    df = contracts.copy()
    # Origin is part of the lane key when a demo contains more than one depot.
    df["lane_group"] = (df["origin_dc"].astype(str) + "|" +
                        df["dest_province"].astype(str) + "|" + df["dest_city"].astype(str))
    df = add_calendar_features(df, "order_date", set(holidays or []))
    return add_lag_rolling_features(df)

def _ordered_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Training encodings use only earlier dates, excluding all same-day targets."""
    df = df.copy()
    for col in ENCODE_COLUMNS:
        daily = df.groupby([col, "order_date"], observed=True, dropna=False)["unit_price"].agg(["sum", "count"]).reset_index()
        daily = daily.sort_values([col, "order_date"])
        grouped = daily.groupby(col, observed=True, dropna=False)
        past_sum = grouped["sum"].cumsum() - daily["sum"]
        past_count = grouped["count"].cumsum() - daily["count"]
        daily[f"{col}_te"] = past_sum / past_count.replace(0, np.nan)
        df = df.merge(daily[[col, "order_date", f"{col}_te"]], on=[col, "order_date"],
                      how="left", validate="many_to_one", sort=False)
    return df

def training_features(history: pd.DataFrame, holidays=None) -> pd.DataFrame:
    return _ordered_encoding(_features(history, holidays))

def query_features(history: pd.DataFrame, queries: pd.DataFrame, holidays=None) -> pd.DataFrame:
    """Fixed-origin features: held-out labels are never used, even within a window."""
    if history.empty or queries.empty:
        raise ValueError("History and queries must both be nonempty.")
    if history["order_date"].max() >= queries["order_date"].min():
        raise ValueError("All history must precede every queried date.")
    h = history.copy()
    q = queries.copy()
    q["unit_price"] = np.nan
    q = q.drop(columns=["freight"], errors="ignore")
    q["_query_order"] = np.arange(len(q))
    h["_query_order"] = np.nan
    out = _features(pd.concat([h, q], ignore_index=True), holidays)
    out = out[out["_query_order"].notna()].sort_values("_query_order").copy()
    for col in ENCODE_COLUMNS:
        table = history.groupby(col, observed=True)["unit_price"].mean()
        out[f"{col}_te"] = out[col].map(table).astype(float).fillna(history["unit_price"].mean())
    return out.reset_index(drop=True)

def compute_metrics(actual, predicted) -> dict[str, Any]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require matching nonempty arrays.")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metrics require finite values.")
    error = np.abs(actual - predicted)
    mask = actual != 0
    relative = error[mask] / np.abs(actual[mask])
    denom = np.abs(actual).sum()
    symmetric_denom = np.abs(actual) + np.abs(predicted)
    symmetric = np.divide(2 * error, symmetric_denom, out=np.zeros_like(error), where=symmetric_denom != 0)
    return {
        "n": int(actual.size),
        "mae": float(error.mean()),
        "mape": float(relative.mean()) if mask.any() else None,
        "wape": float(error.sum() / denom) if denom > 0 else None,
        "smape": float(symmetric.mean()),
        "within_15pct": float((relative <= .15).mean()) if mask.any() else None,
        "n_zero_actual_excluded": int((~mask).sum()),
    }

def validation_split(history: pd.DataFrame, validation_days: int = 28):
    cutoff = history["order_date"].max() - pd.Timedelta(days=validation_days - 1)
    fit = history[history["order_date"] < cutoff].copy()
    valid = history[history["order_date"] >= cutoff].copy()
    if fit.empty or valid.empty:
        raise ValueError("Not enough dated history for an internal validation window.")
    return fit, valid

@dataclass
class ModelBundle:
    model: Any
    categories: dict
    history: pd.DataFrame
    holidays: set
    rounds: int

    def predict(self, queries: pd.DataFrame) -> np.ndarray:
        df = query_features(self.history, queries, self.holidays)
        X = _matrix(df, self.categories)
        return np.maximum(self.model.predict(X), 0.0)

def _category_levels(features):
    return {col: list(pd.Categorical(features[col]).categories) for col in CATEGORIES}

def _matrix(features, levels):
    out = features[FEATURES].copy()
    for col, values in levels.items():
        out[col] = pd.Categorical(out[col], categories=values)
    return out

def _fit(history, rounds, holidays=None, validation=None):
    import lightgbm as lgb
    train = training_features(history, holidays)
    levels = _category_levels(train)
    X = _matrix(train, levels)
    train_set = lgb.Dataset(X, label=train["unit_price"], categorical_feature=CATEGORIES)
    kwargs = {}
    if validation is not None:
        valid = query_features(history, validation, holidays)
        valid_set = lgb.Dataset(_matrix(valid, levels), label=validation["unit_price"].to_numpy(),
                                categorical_feature=CATEGORIES, reference=train_set)
        kwargs = {"valid_sets": [valid_set],
                  "callbacks": [lgb.early_stopping(20, verbose=False)]}
    params = {
        "objective": "regression", "metric": "mae", "learning_rate": .05,
        "num_leaves": 31, "verbosity": -1, "num_threads": 1,
        "seed": 42, "deterministic": True, "force_col_wise": True,
    }
    model = lgb.train(params, train_set, num_boost_round=rounds, **kwargs)
    return ModelBundle(model, levels, history.copy(), set(holidays or []),
                       int(model.best_iteration or rounds))

def fit_forecaster(history, max_rounds=300, holidays=None, validation_days=28):
    """Select rounds on an internal time split, then refit on all pre-test history."""
    fit, valid = validation_split(history, validation_days)
    selected = _fit(fit, max_rounds, holidays, validation=valid).rounds
    return _fit(history, selected, holidays)

def allocate_predictions(lines: pd.DataFrame, contract_predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert predicted contract unit prices to totals and weight-based line charges."""
    if contract_predictions["contract_id"].duplicated().any():
        raise ValueError("Contract predictions must be unique.")
    out = lines.copy()
    out["_line_id"] = np.arange(len(out))
    out = out.merge(contract_predictions[["contract_id", "predicted_unit_price"]],
                    on="contract_id", how="left", validate="many_to_one")
    if out["predicted_unit_price"].isna().any():
        raise ValueError("Every shipment requires a contract prediction.")
    out["predicted_freight"] = out["predicted_unit_price"] * out["weight_ton"]
    return out.sort_values("_line_id").drop(columns="_line_id").reset_index(drop=True)

def rolling_backtest(lines, initial_days=180, test_days=30, max_folds=6,
                     max_rounds=300, holidays=None):
    """Nonoverlapping fixed-origin test windows with expanding training history."""
    if min(initial_days, test_days, max_folds, max_rounds) <= 0:
        raise ValueError("Backtest settings must be positive.")
    lines = validate_shipments(lines)
    contracts = aggregate_contracts(lines)
    start = contracts["order_date"].min() + pd.Timedelta(days=initial_days)
    end = contracts["order_date"].max() + pd.Timedelta(days=1)
    folds, predictions, line_predictions = [], [], []
    for fold in range(max_folds):
        test_start = start + pd.Timedelta(days=fold * test_days)
        test_end = min(test_start + pd.Timedelta(days=test_days), end)
        if test_start >= end:
            break
        history = contracts[contracts["order_date"] < test_start].copy()
        test = contracts[(contracts["order_date"] >= test_start) &
                         (contracts["order_date"] < test_end)].copy()
        if history.empty or test.empty:
            continue
        bundle = fit_forecaster(history, max_rounds, holidays)
        predicted = bundle.predict(test)
        pred_frame = test[["contract_id", "order_date", "unit_price"]].copy()
        pred_frame["predicted_unit_price"] = predicted
        pred_frame["fold"] = fold + 1
        test_lines = lines[lines["contract_id"].isin(test["contract_id"])].copy()
        allocated = allocate_predictions(test_lines, pred_frame)
        allocated["fold"] = fold + 1
        row_metrics = compute_metrics(allocated["freight"] / allocated["weight_ton"],
                                      allocated["predicted_unit_price"])
        folds.append({
            "fold": fold + 1,
            "train_end": (test_start - pd.Timedelta(days=1)).date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_end": (test_end - pd.Timedelta(days=1)).date().isoformat(),
            "n_train_contracts": len(history), "boosting_rounds": bundle.rounds,
            **{f"contract_{k}": v for k, v in compute_metrics(test["unit_price"], predicted).items()},
            **{f"line_{k}": v for k, v in row_metrics.items()},
        })
        predictions.append(pred_frame)
        line_predictions.append(allocated)
    if not folds:
        raise ValueError("No usable test folds; provide more history or reduce initial_days.")
    return pd.DataFrame(folds), pd.concat(predictions, ignore_index=True), pd.concat(line_predictions, ignore_index=True)
