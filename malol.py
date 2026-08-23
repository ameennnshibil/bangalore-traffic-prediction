import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


data = pd.read_csv(r"C:\Users\hp\HTML\iris_dataset.csv3")

print("Dataset:")
print(data.head())


X = data.drop("species", axis=1)
y = data["species"]


encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nEncoded classes:")
print(encoder.classes_)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Sequential()

model.add(Input(shape=(4,)))

model.add(Dense(16, activation="relu"))

model.add(Dense(8, activation="relu"))

model.add(Dense(3, activation="softmax"))


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


print("\nTraining the model...")

model.fit(
    X_train,
    y_train,
    epochs=50,
    verbose=1
)



loss, accuracy = model.evaluate()
    X_test,
    y_test,
    verbose=0


print("\n--------------------------------")
print("Model Evaluation")
print("--------------------------------")

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")