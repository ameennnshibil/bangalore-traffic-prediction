# Deep Feedforward Neural Network - Iris Dataset

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# --------------------------------------------------
# 1. Read the Iris dataset
# --------------------------------------------------

data = pd.read_csv(r"C:\Users\hp\HTML\iris_dataset.csv3")

print("Dataset:")
print(data.head())


# --------------------------------------------------
# 2. Separate input (X) and output (y)
# --------------------------------------------------

X = data.drop("species", axis=1)
y = data["species"]


# --------------------------------------------------
# 3. Convert species names into numbers
# --------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nEncoded classes:")
print(encoder.classes_)


# --------------------------------------------------
# 4. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. Create Deep Feedforward Neural Network
# --------------------------------------------------

model = Sequential()

# Input layer
model.add(Input(shape=(4,)))

# Hidden Layer 1
model.add(Dense(16, activation="relu"))

# Hidden Layer 2
model.add(Dense(8, activation="relu"))

# Output Layer
model.add(Dense(3, activation="softmax"))


# --------------------------------------------------
# 6. Compile the model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 7. Display model structure
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

print("\nTraining the model...")

model.fit(
    X_train,
    y_train,
    epochs=50,
    verbose=1
)


# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n--------------------------------")
print("Model Evaluation")
print("--------------------------------")

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")