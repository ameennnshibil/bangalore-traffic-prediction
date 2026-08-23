import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv("iris_dataset.csv")
print(data.head())
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(4,)))
model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(
 optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
)
history = model.fit(
 X_train,
 y_train,
 epochs=30,
 batch_size=8,
 validation_data=(X_test, y_test),
 verbose=1
)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nFinal Test Loss :", round(loss, 4))
print("Final Test Accuracy :", round(accuracy * 100, 2), "%")
print("\nEpoch-wise Training Results")