# Freight Rate Forecast
**A portfolio adaptation of Susan Wu's internship work on contract-level freight pricing.**

The original business process quotes one truck contract and allocates its freight charge across delivery records by weight. This project models the contract's unit rate, reconstructs its total freight charge, then allocates that total back to delivery rows.

This public version uses **entirely synthetic data** and a generic English schema. No client spreadsheets, identifiers, saved client models, internal reports, or original repository history are included.

## What I worked on
- Examined the pricing process and identified why delivery rows were not independent pricing decisions.
- Changed the modeling grain to contracts, with total weight and farthest delivery distance as a route-distance proxy.
- Built daily lag/rolling features, categorical features, and target encodings for a LightGBM regressor.
- Used expanding-window backtests and compared contract-level and delivery-row metrics.
- Investigated feature contributions and differences among routes in the original project.

## Quick start
Use Python 3.10 or newer.
~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
~~~

On macOS, LightGBM may also require the OpenMP runtime. See the official LightGBM installation guide:
https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html

The demo generates data locally with seed 42 and writes fold metrics, contract predictions, and allocation examples to the results folder. It does not access an external API or load any client files.

## Method
1. Validate shipment rows and consistent contract dates.
2. Aggregate weight and freight within each contract; use maximum delivery distance as a proxy.
3. Build calendar features and 7/30-day lags and rolling means using earlier dates only.
4. Within each outer training window, reserve its last 28 days to choose boosting rounds.
5. Refit on all pre-test history, then evaluate on the next 30 days.
6. Allocate the predicted contract total by weight and report both metric grains.

Test windows do not overlap. A test-window forecast uses information available at its start: actual prices inside that held-out window do not update lag features. Unknown categories are handled without fitting encodings on the test set. Training target encodings exclude all same-day targets.

## Results and interpretation
The files under results contain synthetic demonstration outputs only. They do not reproduce the original project's historical development metrics.

The original project recorded a mean contract-level MAPE of 7.23% and allocated delivery-row MAPE of 7.56% on historical data. Those figures used a different development/backtesting setup, including evaluation windows also used for early stopping. They should not be presented as untouched holdout or production performance. Client data is not distributed, so those historical figures cannot be independently reproduced here.

The public demo separates internal validation from outer testing and freezes target-derived features at each forecast origin. Its results are intentionally reported separately.

## Repository
| File | Purpose |
|---|---|
| freight_forecast.py | Validation, aggregation, encodings, training, allocation, backtesting |
| feature_engineering.py | Reusable calendar and daily lag/rolling calculations adapted from the original pipeline |
| synthetic_data.py | Artificial data generator with generic identifiers |
| run_demo.py | Reproducible demonstration |
| tests/test_pipeline.py | Allocation conservation, timing, metric and model checks |
| docs/methodology.md | Scope, evaluation details, and limitations |
| data/sample_shipments.csv | Small synthetic input example |
| results/ | Generated demonstration results |

## Scope
This forecasts historical freight rates; it does not estimate price elasticity, optimize revenue, or provide real-time market quotes. The farthest-stop distance is a proxy for full-trip distance. A new route or pricing regime requires separate validation.

## Verified synthetic demonstration
The included run used Python 3.12, NumPy 2.3.5, pandas 2.2.3, LightGBM 4.5.0, and SciPy 1.16.3. Exact Python package versions are in requirements-tested.txt. On this Mac, OpenMP was supplied inside the isolated test environment.

| Synthetic demonstration metric | Value |
|---|---|
| Nonoverlapping test folds | 6 |
| Test contract predictions | 1080 |
| Mean contract MAPE across folds | 7.19% |
| Pooled delivery-row MAPE | 7.26% |

Ten automated checks passed, including unchanged predictions when held-out prices are altered, calendar-day lag behavior with missing dates, and preservation of allocated contract charges. See TEST_REPORT.txt. These results describe the artificial demonstration only.

The synthetic demo does not assume a locale-specific public holiday calendar. Library callers may supply an explicit set of holiday dates.
