import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("inventory_stockout_risk_dataset.csv")

# Remove product ID because it is only an identifier
df = df.drop("product_id", axis=1)

# Convert target: No = 0, Yes = 1
df["stockout_risk"] = df["stockout_risk"].map({
    "No": 0,
    "Yes": 1
})

# Separate features and target
X = df.drop("stockout_risk", axis=1)
y = df["stockout_risk"]

# Convert categorical columns into dummy variables
X = pd.get_dummies(
    X,
    columns=["category", "season"],
    drop_first=True
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("========== PREPROCESSING RESULTS ==========")
print("Original dataset:", df.shape)
print("Features after encoding:", X.shape)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nFeature columns:")
print(X.columns.tolist())