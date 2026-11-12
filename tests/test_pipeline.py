"""Check financial allocation, information timing, and held-out label independence."""
import unittest
import numpy as np
import pandas as pd
from freight_forecast import (
    aggregate_contracts, allocate_predictions, compute_metrics, fit_forecaster,
    query_features, training_features, validate_shipments, validation_split,
)
from synthetic_data import make_shipments

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = make_shipments(days=130, contracts_per_day=3)
        cls.contracts = aggregate_contracts(cls.lines)

    def test_aggregation_conserves_totals(self):
        self.assertAlmostEqual(self.lines.freight.sum(), self.contracts.freight.sum())
        self.assertAlmostEqual(self.lines.weight_ton.sum(), self.contracts.weight_ton.sum())
        self.assertTrue(self.contracts.contract_id.is_unique)

    def test_allocation_conserves_contract_total(self):
        predictions = self.contracts[["contract_id"]].copy()
        predictions["predicted_unit_price"] = 200.
        allocated = allocate_predictions(self.lines, predictions)
        by_contract = allocated.groupby("contract_id")["predicted_freight"].sum()
        expected = self.contracts.set_index("contract_id").weight_ton * 200.
        np.testing.assert_allclose(by_contract.sort_index(), expected.sort_index())

    def test_heldout_labels_do_not_change_features(self):
        cutoff = self.contracts.order_date.min() + pd.Timedelta(days=100)
        history = self.contracts[self.contracts.order_date < cutoff]
        query = self.contracts[self.contracts.order_date >= cutoff].copy()
        original = query_features(history, query)
        query["unit_price"] *= 1000
        query["freight"] *= 1000
        changed = query_features(history, query)
        pd.testing.assert_frame_equal(original, changed)

    def test_current_day_labels_do_not_change_own_features(self):
        df = self.contracts.copy()
        date = df.order_date.max()
        first = training_features(df)
        df.loc[df.order_date == date, "unit_price"] *= 1000
        second = training_features(df)
        cols = ["lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30",
                "vehicle_type_te", "customer_type_te"]
        np.testing.assert_allclose(first.loc[first.order_date == date, cols],
                                   second.loc[second.order_date == date, cols],
                                   equal_nan=True)

    def test_queries_cannot_overlap_history(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            query_features(self.contracts, self.contracts.tail(1))

    def test_validation_is_strictly_before_test(self):
        test_start = self.contracts.order_date.max() - pd.Timedelta(days=10)
        history = self.contracts[self.contracts.order_date < test_start]
        train, valid = validation_split(history)
        self.assertLess(train.order_date.max(), valid.order_date.min())
        self.assertLess(valid.order_date.max(), test_start)

    def test_reject_zero_weight_and_conflicting_contract_dates(self):
        broken = self.lines.copy()
        broken.loc[0, "weight_ton"] = 0
        with self.assertRaises(ValueError):
            validate_shipments(broken)
        sample = self.lines.iloc[[0]].copy()
        duplicate = sample.copy()
        duplicate["order_date"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "consistent"):
            aggregate_contracts(pd.concat([sample, duplicate]))

    def test_zero_actual_metric_accounting(self):
        result = compute_metrics([0., 10.], [0., 11.])
        self.assertEqual(result["n_zero_actual_excluded"], 1)
        self.assertAlmostEqual(result["mape"], .1)
        self.assertAlmostEqual(result["smape"], 1/21)
        self.assertIsNone(compute_metrics([0.], [0.])["mape"])

    def test_model_predictions_ignore_future_prices(self):
        cutoff = self.contracts.order_date.min() + pd.Timedelta(days=100)
        history = self.contracts[self.contracts.order_date < cutoff]
        query = self.contracts[self.contracts.order_date >= cutoff].copy()
        bundle = fit_forecaster(history, max_rounds=40)
        first = bundle.predict(query)
        query["freight"] *= 1000
        query["unit_price"] *= 1000
        np.testing.assert_allclose(first, bundle.predict(query))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= 0).all())


    def test_lags_use_calendar_days_despite_missing_dates(self):
        history = self.contracts.iloc[:2].copy()
        for col in ["origin_dc", "dest_province", "dest_city", "customer_type", "vehicle_type"]:
            history[col] = history[col].iloc[0]
        history["order_date"] = pd.to_datetime(["2024-01-01", "2024-01-03"])
        history["unit_price"] = [10., 30.]
        query = history.iloc[[0]].copy()
        query["contract_id"] = "NEW_CONTRACT"
        query["order_date"] = pd.Timestamp("2024-01-08")
        features = query_features(history, query)
        self.assertEqual(features["lag_7"].iloc[0], 10.)
        self.assertEqual(features["rolling_mean_7"].iloc[0], 20.)
        self.assertTrue(pd.isna(features["lag_30"].iloc[0]))

if __name__ == "__main__":
    unittest.main()
