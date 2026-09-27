import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("inventory_stockout_risk_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# 2. REMOVE PRODUCT ID
# =========================================================

df = df.drop("product_id", axis=1)


# =========================================================
# 3. CONVERT TARGET VARIABLE
# =========================================================

df["stockout_risk"] = df["stockout_risk"].map({
    "No": 0,
    "Yes": 1
})


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop("stockout_risk", axis=1)
y = df["stockout_risk"]


# =========================================================
# 5. ENCODE CATEGORICAL VARIABLES
# =========================================================

X = pd.get_dummies(
    X,
    columns=["category", "season"],
    drop_first=True
)


print("\nFeatures after encoding:")
print(X.columns.tolist())


# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 7. CREATE MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}


# =========================================================
# 8. TRAIN AND EVALUATE MODELS
# =========================================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Print metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")


    # Confusion Matrix
    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)


    # Classification Report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Risk",
                "Stockout Risk"
            ],
            zero_division=0
        )
    )


    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    trained_models[name] = model


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results).T

print("\n")
print("=" * 70)
print("IMPROVED MODEL COMPARISON")
print("=" * 70)

print(results_df)


# =========================================================
# 10. DECISION TREE FEATURE IMPORTANCE
# =========================================================

tree_model = trained_models["Decision Tree"]

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": tree_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.to_string(index=False))


# =========================================================
# 11. SAVE TRAINED MODELS
# =========================================================

joblib.dump(
    trained_models["Logistic Regression"],
    "logistic_regression_model.pkl"
)

joblib.dump(
    trained_models["Decision Tree"],
    "decision_tree_model.pkl"
)


# =========================================================
# 12. SAVE FEATURE COLUMNS
# =========================================================

joblib.dump(
    list(X.columns),
    "feature_columns.pkl"
)


# =========================================================
# 13. FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 60)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 60)

print("Created files:")
print("1. logistic_regression_model.pkl")
print("2. decision_tree_model.pkl")
print("3. feature_columns.pkl")