import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. LOAD BANGALORE TRAFFIC DATASET
# =========================================================

data = pd.read_csv("Banglore_traffic_Dataset.csv")

print("========================================")
print("     BANGALORE TRAFFIC PREDICTION")
print("========================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)


# =========================================================
# 2. CONVERT DATE
# =========================================================

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Create useful date features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day


# =========================================================
# 3. CONVERT CONGESTION LEVEL
# =========================================================

# Convert categorical Congestion Level into numerical columns
data = pd.get_dummies(
    data,
    columns=["Congestion Level"],
    drop_first=True
)


# =========================================================
# 4. DEFINE FEATURES
# =========================================================

features = [
    "Average Speed",
    "Travel Time Index",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Year",
    "Month",
    "Day"
]


# Add Congestion Level dummy columns
congestion_columns = [
    column
    for column in data.columns
    if column.startswith("Congestion Level_")
]

features.extend(congestion_columns)


# =========================================================
# 5. DEFINE TARGET
# =========================================================

target = "Traffic Volume"


print("\nFeatures used for prediction:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# =========================================================
# 6. CHECK MISSING VALUES
# =========================================================

print("\n========================================")
print("       MISSING VALUE CHECK")
print("========================================")

print(data[features + [target]].isnull().sum())


# =========================================================
# 7. PREPARE MODEL DATA
# =========================================================

model_data = data[features + [target]].copy()

# Remove rows containing missing values
model_data = model_data.dropna()

print("\nRows available for machine learning:")
print(len(model_data))


# =========================================================
# 8. CREATE X AND Y
# =========================================================

X = model_data[features]
y = model_data[target]


# =========================================================
# 9. SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========================================")
print("          DATASET SPLIT")
print("========================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 10. CREATE MACHINE LEARNING MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 11. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(X_train, y_train)


import joblib
joblib.dump(model, "traffic_model.pkl")

print("Model training completed!")



# =========================================================
# 12. MAKE PREDICTIONS
# =========================================================

predictions = model.predict(X_test)


# =========================================================
# 13. EVALUATE MODEL
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========================================")
print("      TRAFFIC PREDICTION RESULTS")
print("========================================")

print("Actual test values:", len(y_test))

print("Mean Absolute Error:", round(mae, 2))

print("R2 Score:", round(r2, 4))


# =========================================================
# 14. SHOW SAMPLE PREDICTIONS
# =========================================================

results = pd.DataFrame({
    "Actual Traffic Volume": y_test.values,
    "Predicted Traffic Volume": predictions
})

print("\n========================================")
print("        SAMPLE PREDICTIONS")
print("========================================")

print(results.head(10))


# =========================================================
# 15. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("          FEATURE IMPORTANCE")
print("========================================")

print(importance)


# =========================================================
# 16. FINAL MESSAGE
# =========================================================

print("\n========================================")
print("       MODEL TRAINED SUCCESSFULLY!")
print("========================================")
# ============================================================
# 17. VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

print("\n========================================")
print("        TRAFFIC VISUALIZATION")
print("========================================")


# ------------------------------------------------------------
# 1. Actual vs Predicted Traffic Volume
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:100],
    label="Actual Traffic Volume"
)

plt.plot(
    predictions[:100],
    label="Predicted Traffic Volume"
)

plt.title("Bangalore Traffic - Actual vs Predicted")
plt.xlabel("Test Data Samples")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)

plt.savefig("traffic_chart_1.png")
plt.close()


# ------------------------------------------------------------
# 2. Traffic Volume Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    data["Traffic Volume"],
    bins=30
)

plt.title("Bangalore Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Number of Records")
plt.grid(True)

plt.savefig("traffic_chart_2.png")
plt.close()


# ------------------------------------------------------------
# 3. Average Speed Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    data["Average Speed"],
    bins=30
)

plt.title("Average Speed Distribution")
plt.xlabel("Average Speed")
plt.ylabel("Number of Records")
plt.grid(True)

plt.savefig("traffic_chart_3.png")
plt.close()


# ------------------------------------------------------------
# 4. Congestion Level Distribution
# ------------------------------------------------------------

if "Congestion Level" in data.columns:

    plt.figure(figsize=(10, 5))

    data["Congestion Level"].value_counts().plot(
        kind="bar"
    )

    plt.title("Bangalore Traffic Congestion Levels")
    plt.xlabel("Congestion Level")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.close()


# ------------------------------------------------------------
# 5. Feature Importance Graph
# ------------------------------------------------------------

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title("Traffic Prediction - Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.close()


print("\n========================================")
print("       VISUALIZATION COMPLETED")
print("========================================")

# ------------------------------------------------------------
# 2. Traffic Volume Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(data["Traffic Volume"], bins=30)

plt.title("Bangalore Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Number of Records")
plt.grid(True)

plt.close()


# ------------------------------------------------------------
# 3. Average Speed Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(data["Average Speed"], bins=30)

plt.title("Average Speed Distribution")
plt.xlabel("Average Speed")
plt.ylabel("Number of Records")
plt.grid(True)

plt.close()


# ------------------------------------------------------------
# 4. Congestion Level Distribution
# ------------------------------------------------------------

if "Congestion Level" in data.columns:

    plt.figure(figsize=(10, 5))

    data["Congestion Level"].value_counts().plot(
        kind="bar"
    )

    plt.title("Bangalore Traffic Congestion Levels")
    plt.xlabel("Congestion Level")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.close()
print("\n================================")
print("       TRAFFIC PREDICTION SYSTEM")
print("================================")

print("\nEnter traffic details to predict Traffic Volume.")

print("\n==========================================")
print("          TRAFFIC PREDICTION SYSTEM")
print("==========================================")

print("\nEnter traffic details to predict Traffic Volume.")

try:
    average_speed = float(input("Enter Average Speed: "))
    travel_time = float(input("Enter Travel Time Index: "))
    road_capacity = float(input("Enter Road Capacity Utilization: "))

    # Create input using the same columns used by the model
    user_data = {}

    for column in X.columns:
        if column == "Average Speed":
            user_data[column] = average_speed

        elif column == "Travel Time Index":
            user_data[column] = travel_time

        elif column == "Road Capacity Utilization":
            user_data[column] = road_capacity

        else:
            # Use the average value from the dataset
            user_data[column] = data[column].mean()

    user_input = pd.DataFrame([user_data])

    # Make prediction
    user_prediction = model.predict(user_input)

    print("\n==========================================")
    print("             PREDICTION RESULT")
    print("==========================================")

    print(
        "Predicted Traffic Volume:",
        round(user_prediction[0], 2)
    )

    print("==========================================")

except Exception as e:
    print("\nPrediction Error:", e)