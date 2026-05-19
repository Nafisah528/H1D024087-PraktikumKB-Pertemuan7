# Praktikum KB Pertemuan 7 - Jaringan Syaraf Tiruan 2

## Identitas

* Nama : Nafisah Sekar Ayu
* NIM : H1D024087
* Shift Awal: F
* Shift Akhir: C

---

# Deskripsi Praktikum

Pada praktikum ini dilakukan implementasi Jaringan Syaraf Tiruan (JST) menggunakan TensorFlow dan Keras untuk mengklasifikasikan spesies bunga Iris. Dataset Iris dipilih karena merupakan salah satu dataset dasar yang sering digunakan dalam pembelajaran machine learning dan neural network.

Klasifikasi dilakukan berdasarkan empat fitur utama bunga, yaitu:

* panjang sepal
* lebar sepal
* panjang petal
* lebar petal

Program akan mempelajari pola dari data tersebut, kemudian memprediksi jenis bunga Iris ke dalam tiga kelas:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

---

# Dataset

Dataset yang digunakan berasal dari UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/53/iris

---

# Library yang Digunakan

Beberapa library yang digunakan pada praktikum ini antara lain:

* TensorFlow
* Keras
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# Penjelasan Alur Program

## 1. Import Library

Tahap pertama adalah mengimpor library yang diperlukan untuk proses pengolahan data, pembuatan model JST, visualisasi, dan evaluasi model.

```python id="bhkq2n"
import tensorflow as tf
import pandas as pd
import numpy as np
```

TensorFlow dan Keras digunakan untuk membuat neural network, sedangkan Pandas dan NumPy digunakan untuk mengolah data.

---

## 2. Membaca Dataset

Dataset Iris dibaca menggunakan `pandas.read_csv()` dari URL dataset UCI.

```python id="naxp7n"
dataset = pd.read_csv(...)
```

Setelah data berhasil dibaca, data dipisahkan menjadi:

* fitur (X)
* label (y)

Fitur berisi data numerik bunga, sedangkan label berisi nama spesies bunga iris.

---

## 3. Encoding Label

Karena label masih berbentuk teks, label perlu diubah menjadi angka agar dapat diproses oleh model JST.

Contohnya:

* Iris-setosa → 0
* Iris-versicolor → 1
* Iris-virginica → 2

Proses ini dilakukan menggunakan `LabelEncoder`.

```python id="l2q8ks"
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

---

## 4. Membagi Data Training dan Testing

Dataset kemudian dibagi menjadi:

* 80% data training
* 20% data testing

```python id="fsvxlt"
train_test_split(...)
```

Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji performa model setelah proses training selesai.

---

## 5. Membuat Model JST

Model dibuat menggunakan arsitektur Sequential dari Keras.

Model terdiri dari:

* input layer
* beberapa hidden layer
* output layer

```python id="4mkvzb"
Dense(..., activation='relu')
```

Hidden layer menggunakan fungsi aktivasi ReLU agar model dapat mempelajari pola yang lebih kompleks. Pada output layer digunakan fungsi aktivasi Softmax karena kasus ini termasuk klasifikasi multikelas.

---

## 6. Compile Model

Sebelum dilatih, model perlu dikompilasi terlebih dahulu.

```python id="t14ez7"
model.compile(...)
```

Pada tahap ini digunakan:

* optimizer Adam
* loss function sparse categorical crossentropy
* metric accuracy

Optimizer digunakan untuk memperbarui bobot model, sedangkan accuracy digunakan untuk mengukur tingkat ketepatan prediksi model.

---

## 7. Training Model

Model kemudian dilatih menggunakan data training.

```python id="ekv8vo"
model.fit(...)
```

Pada proses training:

* data dimasukkan ke neural network
* model melakukan prediksi
* kesalahan dihitung menggunakan loss function
* bobot diperbarui secara bertahap

Proses ini dilakukan berulang selama beberapa epoch agar model dapat belajar mengenali pola data dengan lebih baik.

---

## 8. Evaluasi Model

Setelah training selesai, model diuji menggunakan data testing.

```python id="hmp0i4"
model.evaluate(...)
```

Hasil evaluasi berupa:

* nilai loss
* nilai accuracy

Semakin tinggi accuracy, maka semakin baik performa model dalam melakukan klasifikasi.

---

## 9. Visualisasi Hasil Training

Grafik accuracy dan loss ditampilkan untuk melihat perkembangan proses training.

Visualisasi ini membantu mengetahui apakah model belajar dengan baik atau mengalami overfitting.

---

## 10. Prediksi Data

Model digunakan untuk memprediksi kelas dari data testing.

```python id="j5jlwm"
model.predict(...)
```

Hasil prediksi berupa probabilitas masing-masing kelas, kemudian dipilih nilai probabilitas tertinggi sebagai hasil klasifikasi akhir.

---

## 11. Confusion Matrix

Confusion matrix digunakan untuk melihat hasil klasifikasi secara lebih detail.

Melalui confusion matrix dapat diketahui:

* jumlah prediksi benar
* jumlah prediksi salah
* performa model pada setiap kelas

---

# Alur Singkat Program

1. Import library
2. Membaca dataset Iris
3. Memisahkan fitur dan label
4. Encoding label
5. Membagi data training dan testing
6. Membuat model JST
7. Compile model
8. Training model
9. Evaluasi model
10. Menampilkan grafik training
11. Prediksi data
12. Menampilkan confusion matrix

---

# Cara Menjalankan Program

Install library:

```bash id="p4z7qw"
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

Menjalankan program:

```bash id="y0vszc"
python pertemuan7.py
```



