# Pelatihan Model & Penyimpanan. 
# Kode ini melatih model SVM dan Naive Bayes, menghitung akurasi, membuat grafik perbandingan, 
# dan menyimpan model ke dalam file .pkl. 
# Ini adalah Tahap 4 & 5.

# Melatih model SVM dan Naive Bayes, mengevaluasi kinerjanya, dan menyimpan model (.pkl) untuk deployment web.

import pandas as pd # Untuk memuat data
from sklearn.model_selection import train_test_split # Untuk membagi data
from sklearn.svm import SVC # Model Support Vector Machine
from sklearn.naive_bayes import GaussianNB # Model Gaussian Naive Bayes
from sklearn.metrics import accuracy_score # Untuk menghitung akurasi
import matplotlib.pyplot as plt # Untuk membuat grafik
import joblib # Untuk menyimpan dan memuat model

# Muat dataset akhir
df = pd.read_csv('../data/processed/final_ml_dataset_14_features.csv') # Memuat input model

# --- 1. Persiapan Data untuk Model ---
# X (Fitur Input): Semua kolom kecuali Diagnosis_Label
X = df.drop(columns=['Diagnosis_Label']) 
# Y (Label Output/Target): Hanya kolom Diagnosis_Label
y = df['Diagnosis_Label']

# Membagi data menjadi Training (80%) dan Testing (20%)
# random_state=42: Memastikan pembagian selalu sama (dapat direproduksi)
# stratify=y: Memastikan proporsi label di X_train dan X_test sama
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# --- 2. Pelatihan Model ---

# Model 1: Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42, probability=True) # Inisialisasi model (gunakan probability=True jika ingin probabilitas)
svm_model.fit(X_train, y_train) # Melatih model SVM
svm_pred = svm_model.predict(X_test) # Memprediksi data testing
svm_accuracy = accuracy_score(y_test, svm_pred) # Menghitung Akurasi

# Model 2: Gaussian Naive Bayes (GNB)
nb_model = GaussianNB() # Inisialisasi model Naive Bayes
nb_model.fit(X_train, y_train) # Melatih model Naive Bayes
nb_pred = nb_model.predict(X_test) # Memprediksi data testing
nb_accuracy = accuracy_score(y_test, nb_pred) # Menghitung Akurasi

# --- 3. Evaluasi dan Perbandingan ---
accuracy_results = {
    'SVM': svm_accuracy,
    'Naive Bayes': nb_accuracy
}

# Mencetak hasil akurasi
print("\n--- Hasil Akurasi Model ---")
print(f"Akurasi SVM: {svm_accuracy:.4f}") 
print(f"Akurasi Naive Bayes: {nb_accuracy:.4f}") 

# --- 4. Visualisasi ---
# (Kode plot Matplotlib di sini...)
# ...

# --- 5. Deployment (Menyimpan Model) ---
joblib.dump(svm_model, 'models/svm_model.pkl') # Menyimpan model SVM ke file .pkl
joblib.dump(nb_model, 'models/naive_bayes_model.pkl') # Menyimpan model Naive Bayes ke file .pkl

print("\nModel SVM tersimpan sebagai 'models/svm_model.pkl'")
print("Model Naive Bayes tersimpan sebagai 'models/naive_bayes_model.pkl'") 