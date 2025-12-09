# Pembuatan Label Diagnosis. Kode ini memuat data PHQ-9, 
# menghitung Skor Total, dan membuat kolom Diagnosis_Label (target output: 0, 1, 2, 3).
# Ini adalah Tahap 1.

# Pembersihan data PHQ-9, perhitungan Skor Total, dan pembuatan Label Diagnosis (Variabel Target Y).

import pandas as pd # Import library Pandas untuk manipulasi data tabular (DataFrame)
import numpy as np # Import library NumPy untuk operasi matematika numerik
import os

# Kolom PHQ-9 yang dikonfirmasi (9 Fitur Inti)
PHQ_COLUMNS = [ # Daftar 9 kolom kuesioner PHQ-9
    'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 
    'phq6', 'phq7', 'phq8', 'phq9' 
]

file_phq9 = '../data/raw/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv'# Mendefinisikan nama file sumber

# Muat data
df_phq9 = pd.read_csv(file_phq9) # Membaca file CSV ke dalam Pandas DataFrame

# --- LANGKAH 1: Pembersihan Awal dan Konversi Tipe Data ---
# Konversi kolom PHQ-9 ke numerik. errors='coerce' akan mengubah data non-angka menjadi NaN
for col in PHQ_COLUMNS:
    df_phq9[col] = pd.to_numeric(df_phq9[col], errors='coerce') 

# Menghapus baris yang tidak memiliki jawaban lengkap (NaN) di 9 kolom PHQ-9
df_phq9.dropna(subset=PHQ_COLUMNS, inplace=True) 

# --- LANGKAH 2: Hitung Skor Total PHQ-9 ---
# Menjumlahkan skor 9 kolom PHQ-9 secara horizontal (axis=1) untuk setiap baris
df_phq9['Total_Score'] = df_phq9[PHQ_COLUMNS].sum(axis=1) 

# --- LANGKAH 3: Buat Label Diagnosis (Target Output) ---
def classify_depression(score): # Mendefinisikan fungsi klasifikasi sesuai aturan klinis
    if score <= 4:
        return 0  # Risiko Minimal
    elif 5 <= score <= 9:
        return 1  # Risiko Ringan (Mild)
    elif 10 <= score <= 14:
        return 2  # Risiko Sedang (Moderate)
    else: 
        return 3  # Risiko Berat (Severe)

# Menerapkan fungsi klasifikasi ke kolom Total_Score untuk membuat kolom Diagnosis_Label
df_phq9['Diagnosis_Label'] = df_phq9['Total_Score'].apply(classify_depression) 

# Memilih kolom yang diperlukan untuk langkah penggabungan (user_id, fitur PHQ-9, dan Label)
final_cols = ['user_id'] + PHQ_COLUMNS + ['Diagnosis_Label']
df_phq9_final = df_phq9[final_cols]

# Menyimpan hasil proses ini (Hasil Tahap 1)
df_phq9_final.to_csv('phq9_processed.csv', index=False)