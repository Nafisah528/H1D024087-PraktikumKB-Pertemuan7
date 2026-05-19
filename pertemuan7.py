# PRAKTIKUM 7 - JARINGAN SYARAF TIRUAN 2
# KLASIFIKASI DATASET IRIS

# 1. IMPORT LIBRARY
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# 2. LOAD DATASET IRIS DARI UCI
dataset = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None
)

# Menampilkan 5 data pertama
print(dataset.head())


# 3. MEMISAHKAN FITUR (X) DAN LABEL (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("\nFitur:")
print(X[:5])

print("\nLabel:")
print(y[:5])


# 4. ENCODING LABEL
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nLabel setelah encoding:")
print(y[:5])


# 5. MEMBAGI DATA TRAINING DAN TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))


# 6. MEMBUAT MODEL NEURAL NETWORK
model = Sequential([
    Input(shape=X_train.shape[1:]),

    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),

    Dense(3, activation='softmax')
])

# Menampilkan arsitektur model
model.summary()


# 7. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 8. TRAINING MODEL
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# 9. EVALUASI MODEL
loss, accuracy = model.evaluate(X_test, y_test)

print("\nLoss :", loss)
print("Accuracy :", accuracy)


# 10. VISUALISASI ACCURACY DAN LOSS
pd.DataFrame(history.history).plot(figsize=(10,6))

plt.title("Grafik Training")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.show()


# 11. PREDIKSI DATA TESTING
predictions = model.predict(X_test)

# Mengambil indeks probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

print("\nHasil Prediksi:")
print(predicted_classes)

print("\nLabel Asli:")
print(y_test)


# 12. CONFUSION MATRIX
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.show()


# 13. PREDIKSI DATA BARU
def predict_new_data():

    sepal_length = float(input("Masukkan sepal length : "))
    sepal_width = float(input("Masukkan sepal width  : "))
    petal_length = float(input("Masukkan petal length : "))
    petal_width = float(input("Masukkan petal width  : "))

    new_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(new_data)

    predicted_class = prediction.argmax(axis=1)

    predicted_label = label_encoder.inverse_transform(predicted_class)

    print("\nPrediksi kelas :", predicted_label[0])


# Memanggil fungsi prediksi
predict_new_data()