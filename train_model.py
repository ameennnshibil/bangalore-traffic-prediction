import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


base_folder = Path(__file__).resolve().parent


csv_files = list(base_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the project folder."
    )

csv_file = csv_files[0]

print("Dataset found:")
print(csv_file)
print()


data = pd.read_csv(csv_file)

print("Dataset loaded successfully.")
print()

print("Dataset columns:")
print(data.columns.tolist())
print()


required_columns = [
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"Required column is missing: {column}"
        )


data["Traffic Delay"] = (
    data["Travel Time Index"] - 1
) * 100


if "Road Capacity Utilization" in data.columns:

    data["Road Usage"] = (
        data["Road Capacity Utilization"]
    )

    if data["Road Usage"].max() <= 1:
        data["Road Usage"] = (
            data["Road Usage"] * 100
        )

else:

    maximum_volume = (
        data["Traffic Volume"].max()
    )

    data["Road Usage"] = (
        data["Traffic Volume"]
        / maximum_volume
    ) * 100


data = data[
    [
        "Average Speed",
        "Traffic Delay",
        "Road Usage",
        "Traffic Volume"
    ]
]


data = data.dropna()


data = data[
    (data["Average Speed"] >= 0)
    &
    (data["Traffic Delay"] >= 0)
    &
    (data["Road Usage"] >= 0)
    &
    (data["Road Usage"] <= 100)
    &
    (data["Traffic Volume"] >= 0)
]


print("Training data prepared.")
print()

print(
    "Number of records:",
    len(data)
)

print()

print("Traffic Delay range:")
print(
    f"{data['Traffic Delay'].min():.2f}% "
    f"to "
    f"{data['Traffic Delay'].max():.2f}%"
)

print()

print("Road Usage range:")
print(
    f"{data['Road Usage'].min():.2f}% "
    f"to "
    f"{data['Road Usage'].max():.2f}%"
)

print()

print("Traffic Volume range:")
print(
    f"{data['Traffic Volume'].min():.0f} "
    f"to "
    f"{data['Traffic Volume'].max():.0f}"
)

print()


X = data[
    [
        "Average Speed",
        "Traffic Delay",
        "Road Usage"
    ]
]


y = data["Traffic Volume"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training model...")


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Model training completed.")
print()


predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


print(
    f"Mean Absolute Error: {mae:.2f}"
)

print()


model_file = (
    base_folder / "traffic_model.pkl"
)


joblib.dump(
    model,
    model_file
)


print("New traffic_model.pkl created successfully.")
print()

print("Model location:")
print(model_file)

print()

print("Model uses these 3 inputs:")
print("1. Average Speed")
print("2. Traffic Delay")
print("3. Road Usage")

print()

print("Training finished successfully.")