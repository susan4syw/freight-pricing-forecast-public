# Methodology and provenance

## Public adaptation
Author: Zijing (Susan) Wu.

The pipeline is adapted from the author's internship project. The contract aggregation rule, calendar and lag calculations, target-encoding approach, LightGBM workflow, and dual-grain evaluation follow the original work. Only reusable code functions were extracted; the private repository, client files, model binaries, analysis exports, and internal instructions were not copied.

Changes for this public demonstration:
- English schema and generic depot, route, and customer identifiers.
- Artificial records generated from an illustrative formula, not resampled client records or client-fitted distributions.
- Internal chronological validation for early stopping, separate from outer test windows.
- Fixed-origin features: held-out window prices cannot feed later held-out feature rows.
- Earlier-date-only encodings for training rows and train-only mappings for queries.
- Explicit checks for inconsistent contract dates, invalid weights, and zero-denominator metrics.

These changes make the demonstration self-contained; they do not retroactively change the original historical results.

## Target and allocation
For each contract:
- actual unit rate = total freight / total weight
- predicted contract freight = predicted unit rate * total weight
- predicted delivery freight = predicted contract freight * delivery weight / total weight

As a result, the predicted unit rate is the same across deliveries in a contract, while the allocated freight amount depends on each delivery's weight.

Contract MAPE weights contracts equally. Delivery-row MAPE weights delivery rows equally. These are different summaries and should not be compared as though they had the same denominator. Fold-mean MAPE also differs from pooled MAPE when fold sizes differ.

## Information timing
Features assume order date, planned route, vehicle type, customer category, distance, and weight are known when a quote is prepared. This assumption must be checked for any new application.

Daily target lags are built on a complete calendar within each route. Seven days means seven calendar days, not seven observed records. Rolling averages exclude the current date. Missing history remains missing.

At each outer test origin, all held-out targets are hidden for feature construction. Training selects boosting rounds using only a preceding internal validation period, then refits on the full outer training history. No outer-test score is used to choose boosting rounds.

## Synthetic data
The generator creates generic routes and contracts, with freight proportional to each stop's share of total weight. Artificial rates depend on distance, weight, vehicle category, customer category, a mild time trend, and random noise. This is a demonstration of the workflow, not an empirical model of a freight market.

The included synthetic dataset and scores do not validate performance on real contracts.

## Limitations
- A fixed 30-day origin does not incorporate quote outcomes as they arrive inside the test month.
- Farthest-stop distance is not optimized truck-route distance.
- Ordered encodings have little history early in a series.
- No geographic transferability, causal pricing effects, business savings, or production deployment is claimed.
- The original exploratory backtests used their scoring windows for early stopping. Their reported metrics must retain that qualification.
