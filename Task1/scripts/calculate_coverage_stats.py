import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Command-line arguments
input_file = sys.argv[1]
output_stats = sys.argv[2]
output_plot = sys.argv[3]

# Load the methylation data
data = pd.read_csv(input_file)

# Summing numeric columns to compute Coverage
numeric_columns = ["`000", "`001", "`010", "`011", "`100", "`101", "`110", "`111"]
data["Coverage"] = data[numeric_columns].sum(axis=1)

# Check the new column
if "Coverage" not in data.columns or not pd.api.types.is_numeric_dtype(
    data["Coverage"]
):
    raise ValueError("Failed to compute 'Coverage' column.")

# Group by Tissue and calculate median and CV
stats = data.groupby("Tissue")["Coverage"].agg(["median", "std"])
stats["CV"] = stats["std"] / stats["median"]

# Save statistics to a CSV file
stats.to_csv(output_stats)
print(f"Coverage statistics saved to: {output_stats}")

# Plot coverage distribution by tissue
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x="Tissue", y="Coverage")
plt.title("Coverage Distribution by Tissue")
plt.xlabel("Tissue")
plt.ylabel("Coverage")
plt.tight_layout()
plt.savefig(output_plot)
print(f"Coverage plot saved to: {output_plot}")
