"""Synthetic donor data generator for NayePankh ML mini project."""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 1000

data = {
    "pages_visited": np.random.randint(1, 15, N_SAMPLES),
    "time_on_site_sec": np.random.randint(30, 600, N_SAMPLES),
    "mobile_user": np.random.choice([0, 1], N_SAMPLES, p=[0.6, 0.4]),
    "viewed_campaign": np.random.choice([0, 1], N_SAMPLES, p=[0.5, 0.5]),
    "city_kanpur": np.random.choice([0, 1], N_SAMPLES, p=[0.4, 0.6]),
    "from_instagram": np.random.choice([0, 1], N_SAMPLES, p=[0.55, 0.45]),
}

df = pd.DataFrame(data)

donate_prob = (
    0.05
    + df["pages_visited"] * 0.02
    + df["viewed_campaign"] * 0.15
    + df["time_on_site_sec"] / 2000
    - df["mobile_user"] * 0.05
    + df["from_instagram"] * 0.08
)
donate_prob = donate_prob.clip(0, 0.95)
df["donated"] = (np.random.random(N_SAMPLES) < donate_prob).astype(int)

os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "donor_data.csv")
df.to_csv(output_path, index=False)

print(f"Data ban gaya! File: {output_path}")
print(f"Total rows: {len(df)}")
print(f"Donors: {df['donated'].sum()} | Non-donors: {(df['donated'] == 0).sum()}")
print(f"Donation rate: {df['donated'].mean() * 100:.1f}%")
print("\nSample data:")
print(df.head())
