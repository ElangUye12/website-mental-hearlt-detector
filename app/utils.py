# app/utils.py
import pandas as pd
import numpy as np 

FEATURE_COLUMNS = [
    'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
    'Age', 'family_history', 'treatment', 'work_interfere', 'Gender'
]

def preprocess_input(input_data):
    if not isinstance(input_data, dict):
        raise ValueError("Input harus berupa objek JSON.")

    # Validasi dan persiapan data
    df = pd.DataFrame([input_data])

    # --- Validasi Data Numerik (PHQ & Age) ---
    for col in ['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9','Age']:
        if col not in df.columns:
            raise ValueError(f"Kolom numerik hilang: '{col}'")
        try:
            # Pastikan konversi ke numerik
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            if df[col].isnull().iloc[0]:
                 raise ValueError()
        except:
            raise ValueError(f"Nilai tidak valid atau kosong untuk: '{col}'")

    # --- Encoding Binary/Categorical sesuai dengan proses training ---

    # family_history & treatment (Yes=1, No=0)
    fh_map = {'Yes': 1, 'No': 0}
    try:
        fh_val = str(df['family_history'].iloc[0]).strip()
        if fh_val not in fh_map: raise ValueError()
        df['family_history'] = df['family_history'].map(fh_map).astype(int)
    except:
         raise ValueError("Nilai tidak valid untuk 'family_history'")
    
    tr_map = {'Yes': 1, 'No': 0}
    try:
        tr_val = str(df['treatment'].iloc[0]).strip()
        if tr_val not in tr_map: raise ValueError()
        df['treatment'] = df['treatment'].map(tr_map).astype(int)
    except:
         raise ValueError("Nilai tidak valid untuk 'treatment'")

    # work_interfere (Ordinal: Never=0, Rarely=1, Sometimes=2, Often=3)
    wi_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    wi_val = str(df['work_interfere'].iloc[0]).strip()
    if wi_val not in wi_map:
        raise ValueError(f"Nilai tidak valid untuk 'work_interfere'")
    df['work_interfere'] = df['work_interfere'].map(wi_map).astype(int)

    # Gender (Binary: Male=1, Female/Other=0)
    g = str(df['Gender'].iloc[0]).strip().lower()
    df['Gender'] = 1 if g in ['male', 'm', 'laki-laki', 'pria'] else 0

    # Memastikan urutan kolom input sama persis
    final_input = df[FEATURE_COLUMNS].values
    
    if final_input.shape != (1, len(FEATURE_COLUMNS)):
         raise ValueError(f"Ukuran input salah.")

    return final_input