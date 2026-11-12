"""Generate artificial contracts; no client records or fitted client parameters are used."""
from __future__ import annotations
import numpy as np
import pandas as pd

def make_shipments(days=360, contracts_per_day=6, seed=42):
    if days < 1 or contracts_per_day < 1:
        raise ValueError("days and contracts_per_day must be positive.")
    rng = np.random.default_rng(seed)
    records = []
    counter = 0
    for day_index, day in enumerate(pd.date_range("2024-01-01", periods=days)):
        for _ in range(contracts_per_day):
            counter += 1
            n_stops = int(rng.integers(1, 4))
            total_weight = float(rng.uniform(4, 28))
            distance = float(rng.uniform(100, 1300))
            vehicle = str(rng.choice(["Small", "Large"]))
            customer = str(rng.choice(["Type_A", "Type_B"]))
            lane = int(rng.integers(1, 5))
            rate = (100 + .16 * distance + 220 / np.sqrt(total_weight))
            rate *= (1.13 if vehicle == "Small" else 1.0) * (1.04 if customer == "Type_B" else 1.0)
            rate *= (1 + .04 * np.sin(day_index / 365 * 2 * np.pi) + .0003 * day_index)
            rate *= float(rng.lognormal(mean=0, sigma=.08))
            shares = rng.dirichlet(np.ones(n_stops))
            for stop, share in enumerate(shares):
                weight = total_weight * float(share)
                records.append({
                    "contract_id": f"DEMO_{counter:06d}", "order_date": day,
                    "origin_dc": "Depot_A", "dest_province": f"Region_{lane}",
                    "dest_city": f"Hub_{lane}", "customer_type": customer,
                    "vehicle_type": vehicle,
                    "distance_km": distance if stop == n_stops - 1 else distance * float(rng.uniform(.3, .9)),
                    "weight_ton": weight, "freight": rate * weight,
                })
    return pd.DataFrame(records)
