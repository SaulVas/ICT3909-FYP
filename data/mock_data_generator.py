import random
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define number of rows (random between 100-200)
num_rows = random.randint(100, 200)

# Define discrete steps for the sail settings
mainsheet_lengths = np.linspace(0.5, 3.0, 6)  # 6 discrete values between 0.5 and 3.0
jibsheet_lengths = np.linspace(0.5, 2.5, 5)  # 5 discrete values between 0.5 and 2.5
cunningham_lengths = np.linspace(0.2, 1.5, 4)  # 4 discrete values between 0.2 and 1.5
outhaul_lengths = np.linspace(0.3, 2.0, 5)  # 5 discrete values between 0.3 and 2.0
vang_lengths = np.linspace(0.4, 2.2, 5)  # 5 discrete values between 0.4 and 2.2
traveller_lengths = np.linspace(-0.5, 0.5, 7)  # 7 discrete values between -0.5 and 0.5


# Function to generate TCD values for each row
def generate_tcd_values(n_rows):
    tcd_low = []
    tcd_med = []
    tcd_high = []

    for _ in range(n_rows):
        # Generate 3 values for each TCD level: one angle (0-30) and two floats (0-1)
        tcd_low.append(
            [
                round(random.uniform(0, 30), 2),
                round(random.uniform(0, 1), 2),
                round(random.uniform(0, 1), 2),
            ]
        )

        tcd_med.append(
            [
                round(random.uniform(0, 30), 2),
                round(random.uniform(0, 1), 2),
                round(random.uniform(0, 1), 2),
            ]
        )

        tcd_high.append(
            [
                round(random.uniform(0, 30), 2),
                round(random.uniform(0, 1), 2),
                round(random.uniform(0, 1), 2),
            ]
        )

    return tcd_low, tcd_med, tcd_high


# Generate TCD values
tcd_low_values, tcd_med_values, tcd_high_values = generate_tcd_values(num_rows)

# Generate data
data = {
    "sog": np.random.uniform(1.0, 15.0, num_rows),  # Speed Over Ground (knots)
    "twd": np.random.uniform(0.0, 359.99, num_rows),  # True Wind Direction (degrees)
    "cog": np.random.uniform(0.0, 359.99, num_rows),  # Course Over Ground (degrees)
    "tws": np.random.uniform(2.0, 25.0, num_rows),  # True Wind Speed (knots)
    "mainsheet_length": np.random.choice(mainsheet_lengths, num_rows),
    "jibsheet_length": np.random.choice(jibsheet_lengths, num_rows),
    "cunningham_length": np.random.choice(cunningham_lengths, num_rows),
    "main_halyard_length": np.full(num_rows, 1.5),  # Constant value of 1.5
    "jib_halyard_length": np.full(num_rows, 1.0),  # Constant value of 1.0
    "outhaul_length": np.random.choice(outhaul_lengths, num_rows),
    "vang_length": np.random.choice(vang_lengths, num_rows),
    "traveller_length": np.random.choice(traveller_lengths, num_rows),
    "tcd_low": tcd_low_values,
    "tcd_med": tcd_med_values,
    "tcd_high": tcd_high_values,
    "image_name": [f"mainsail_{i}.jpg" for i in range(num_rows)],
}
# Create DataFrame
df = pd.DataFrame(data)

# Round floating point values to 2 decimal places for readability
df = df.round(2)

# Save to CSV
df.to_csv("data/mock_dataset.csv", index=False)

print(f"Generated {num_rows} rows of mock sailing data in data/mock_dataset.csv")
print("Data ranges:")
for column in df.columns:
    print(f"{column}: min={df[column].min()}, max={df[column].max()}")
