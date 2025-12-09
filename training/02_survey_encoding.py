# Pemrosesan Fitur Pelengkap. 
# Kode ini memuat survey.csv, membersihkan 5 fitur pelengkap, dan mengubah data teks (kategori) menjadi angka (Encoding). 
# Ini adalah Tahap 2.

# Mengambil 5 fitur pelengkap dari survey.csv, membersihkan data, dan melakukan Encoding (mengubah teks menjadi angka).

import pandas as pd # Import Pandas
import numpy as np # Import NumPy

# Daftar 5 Kolom Pelengkap yang dipilih dari data Survey
SURVEY_COLUMNS = [ 
    'Age', 'Gender', 'family_history', 'treatment', 'work_interfere'
]
file_survey = '../data/raw/survey.csv' 

# Muat data survey
df_survey = pd.read_csv(file_survey) 
df_features = df_survey[SURVEY_COLUMNS].copy() # Membuat salinan hanya dengan 5 kolom ini

# --- 1. Pembersihan & Konversi 'Age' ---
df_features['Age'] = pd.to_numeric(df_features['Age'], errors='coerce') # Konversi Usia ke angka
df_features.dropna(subset=['Age'], inplace=True) # Hapus baris jika Usia tidak valid/hilang
df_features = df_features[(df_features['Age'] >= 18) & (df_features['Age'] <= 99)] # Memfilter usia di rentang 18-99

# --- 2. Encoding Kolom Kategori ---
# family_history & treatment (Encoding Biner: Yes=1, No=0)
for col in ['family_history', 'treatment']:
    df_features[col] = df_features[col].map({'Yes': 1, 'No': 0}) # Mapping Yes/No ke 1/0
    # Mengisi nilai hilang (NaN) dengan nilai yang paling sering (mode)
    df_features[col].fillna(df_features[col].mode()[0], inplace=True) 

# work_interfere (Encoding Ordinal: Skala 0-3)
mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3} # Mapping dari teks ke angka
df_features['work_interfere'] = df_features['work_interfere'].map(mapping) 
df_features['work_interfere'].fillna(df_features['work_interfere'].mode()[0], inplace=True) # Mengisi data hilang

# Gender (Encoding Biner: Male=1 vs Female/Other=0)
def simplify_gender(g): # Mendefinisikan fungsi untuk menyederhanakan Gender
    g = str(g).lower().strip()
    if 'male' in g or g in ['m', 'male', 'cis male', 'm.']:
        return 1 # 1 = Laki-laki
    else:
        return 0 # 0 = Perempuan/Lainnya

df_features['Gender'] = df_features['Gender'].apply(simplify_gender) # Menerapkan fungsi Gender
df_features.rename(columns={'Gender': 'Gender_Male'}, inplace=True) # Ganti nama kolom (jika diperlukan)
df_features.reset_index(drop=True, inplace=True) # Mereset index untuk penggabungan

# Menyimpan fitur pelengkap yang sudah di-encode (Hasil Tahap 2)
df_features.to_csv('survey_processed_features.csv', index=False)