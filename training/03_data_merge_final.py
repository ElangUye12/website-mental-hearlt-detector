# Penggabungan Data Final. 
# Kode ini memuat hasil dari langkah 1 dan 2, menyelaraskan baris, 
# dan menggabungkannya menjadi file final_ml_dataset_14_features.csv. 
# Ini adalah Tahap 3.

# Menggabungkan data PHQ-9 yang berlabel dengan 5 fitur pelengkap yang sudah di-encode menjadi dataset final untuk ML.

import pandas as pd # Import Pandas

# Muat hasil dari Tahap 1 dan Tahap 2
df_phq9 = pd.read_csv('phq9_processed.csv') # Data PHQ-9 + Label
df_survey_features = pd.read_csv('survey_processed_features.csv') # Data Survey yang sudah di-encode

# --- 1. Menyelaraskan Ukuran Dataset ---
# Mengambil jumlah baris terkecil untuk penggabungan yang aman
min_rows = min(len(df_phq9), len(df_survey_features))

# Mengambil subset baris (head(min_rows)) dari kedua DataFrame
df_phq9_subset = df_phq9.head(min_rows).drop(columns=['user_id'], errors='ignore') # Menghapus user_id
df_survey_subset = df_survey_features.head(min_rows).reset_index(drop=True)
# Menghapus kolom index temporer dari survey (jika ada)
df_survey_subset.drop(columns=[col for col in df_survey_subset.columns if 'temp_id' in col], inplace=True, errors='ignore') 

# --- 2. Menggabungkan Data (Total 14 Fitur + 1 Label) ---
# Menggabungkan kedua DataFrame secara horizontal (axis=1)
df_final_data = pd.concat([df_phq9_subset.reset_index(drop=True), 
                            df_survey_subset], axis=1)

# Mengganti nama kolom Gender_Male menjadi Gender (untuk input ke model)
df_final_data.rename(columns={'Gender_Male': 'Gender'}, inplace=True)

# --- 3. Finalisasi dan Penyimpanan Dataset ML ---
# Menyimpan dataset lengkap yang siap untuk pelatihan (Hasil Tahap 3)
df_final_data.to_csv('final_ml_dataset_14_features.csv', index=False)