import os
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Create output directories
os.makedirs("tables", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Path to logs
log_dir = "logs"
log_files = sorted(glob.glob(os.path.join(log_dir, "dropout_*.log")))

# Initialize dictionaries
train_data = {}
valid_data = {}

# Read each log file
for log_file in log_files:
    dropout = os.path.basename(log_file).replace("dropout_", "").replace(".log", "")
    train_data[dropout] = []
    valid_data[dropout] = []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["epoch"] == "test":
                continue
            train_data[dropout].append(float(row["train_ppl"]))
            valid_data[dropout].append(float(row["valid_ppl"]))

# Convert to pandas DataFrames
train_df = pd.DataFrame(train_data)
valid_df = pd.DataFrame(valid_data)

train_df.index += 1  # Start epoch numbering from 1
valid_df.index += 1

# Save tables
train_df.to_csv("tables/train_perplexity.csv")
valid_df.to_csv("tables/valid_perplexity.csv")

# Plot
os.makedirs("plots", exist_ok=True)

plt.figure()
train_df.plot(title="Training Perplexity by Dropout", xlabel="Epoch", ylabel="Perplexity")
plt.grid(True)
plt.savefig("plots/train_perplexity.png")

plt.figure()
valid_df.plot(title="Validation Perplexity by Dropout", xlabel="Epoch", ylabel="Perplexity")
plt.grid(True)
plt.savefig("plots/valid_perplexity.png")

print("Tables saved to 'tables/', plots saved to 'plots/'")