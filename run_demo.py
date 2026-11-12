"""Run the public demonstration and save reproducible, explicitly synthetic results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from freight_forecast import compute_metrics, rolling_backtest
from synthetic_data import make_shipments

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    # The demo always generates artificial data; it never discovers or loads client files.
    lines = make_shipments(seed=args.seed)
    folds, predictions, allocated = rolling_backtest(
        lines, max_folds=args.max_folds, max_rounds=args.rounds)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_source": "synthetic_only",
        "seed": args.seed,
        "evaluation": "fixed-origin expanding-window; internal validation for early stopping",
        "test_window_days": 30,
        "n_folds": len(folds),
        "n_contract_predictions": len(predictions),
        "contract_mape_mean_over_folds": float(folds["contract_mape"].mean()),
        "contract_pooled_metrics": compute_metrics(predictions["unit_price"], predictions["predicted_unit_price"]),
        "line_pooled_metrics": compute_metrics(allocated["freight"] / allocated["weight_ton"],
                                              allocated["predicted_unit_price"]),
        "note": "Synthetic demonstration results are not the historical client backtest results.",
    }
    (args.output / "demo_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    folds.to_csv(args.output / "backtest_folds.csv", index=False)
    predictions.to_csv(args.output / "contract_predictions.csv", index=False)
    allocated.head(50).to_csv(args.output / "allocation_examples.csv", index=False)
    (Path("data") / "sample_shipments.csv").parent.mkdir(exist_ok=True)
    lines.head(50).to_csv(Path("data") / "sample_shipments.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
