import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2


# Command-line arguments
input_file = sys.argv[1]  # Input CSV file
output_file = sys.argv[2]  # Output file to save selected biomarkers

# Load the dataset
data = pd.read_csv(input_file)

# Ensure necessary columns are present
required_columns = [
    "strand",
    "CpG_Coordinates",
    "`000",
    "`001",
    "`010",
    "`011",
    "`100",
    "`101",
    "`110",
    "`111",
    "Tissue",
]
if not set(required_columns).issubset(data.columns):
    raise KeyError(f"Required columns {required_columns} are not in the dataset.")

# Combine strand and CpG_Coordinates to create unique PMP identifier
data["PMP"] = data["strand"] + ":" + data["CpG_Coordinates"]

# Select methylation status columns as features
methylation_columns = ["`000", "`001", "`010", "`011", "`100", "`101", "`110", "`111"]
X = data[methylation_columns]

# Encode the target variable (Tissue)
le = LabelEncoder()
y = le.fit_transform(data["Tissue"])  # Convert Tissue categories to numerical labels

# Split the data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature selection: Select top 5 most relevant features using Chi-squared test (to speed up)
selector = SelectKBest(score_func=chi2, k=5)  # Reduce number of features to 5
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train a classifier to predict Tissue differentiation (using fewer trees in Random Forest for speed)
clf = RandomForestClassifier(
    n_estimators=50, random_state=42
)  # Reduce the number of trees
clf.fit(X_train_selected, y_train)

# Make predictions and calculate precision and ROC AUC for Tissue #1 differentiation
y_pred = clf.predict(X_test_selected)
precision = precision_score(
    y_test, y_pred, pos_label=0
)  # Assuming Tissue #1 is encoded as 0
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test_selected)[:, 1])

print(f"Precision for Tissue #1 differentiation: {precision}")
print(f"ROC AUC for Tissue differentiation: {roc_auc}")

# Calculate the mean variant read fraction (VRF) for each PMP in both tissues
# We assume the variant read fraction (VRF) is the average methylation status across the columns
data["VRF"] = data[methylation_columns].mean(axis=1)

# Group by Tissue and calculate the mean VRF for each tissue type
mean_vrf = data.groupby("Tissue")["VRF"].mean()
print(f"Mean VRF for each tissue:\n{mean_vrf}")

# Identify important biomarkers (PMPs) using feature importance from the classifier
importances = clf.feature_importances_
sorted_idx = importances.argsort()[::-1]  # Sort by importance in descending order
top_biomarkers = [methylation_columns[i] for i in sorted_idx[:3]]  # Top 3 biomarkers

print(
    f"Top 3 Biomarkers for Tissue differentiation (Based on feature importance): {top_biomarkers}"
)

# Save selected biomarkers and mean VRF to an output file
output_df = pd.DataFrame(
    {
        "Selected_Biomarkers": top_biomarkers,
        "Mean_VRF_Tissue_1": mean_vrf.get(
            0, "N/A"
        ),  # assuming Tissue #1 is encoded as 0
        "Mean_VRF_Tissue_2": mean_vrf.get(
            1, "N/A"
        ),  # assuming Tissue #2 is encoded as 1
    }
)

output_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")
