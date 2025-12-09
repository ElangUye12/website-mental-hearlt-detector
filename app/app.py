from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- 1. MEMUAT MODEL YANG SUDAH DILATIH ---
# Jalur ke model (sesuaikan dengan struktur folder: folder_proyek/models/nama_file.pkl)
try:
    SVM_MODEL = joblib.load('models/svm_model.pkl')
    NB_MODEL = joblib.load('models/naive_bayes_model.pkl')
    # Daftar 14 fitur yang sama persis saat training
    FEATURE_COLUMNS = [
        'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
        'Age', 'family_history', 'treatment', 'work_interfere', 'Gender'
    ]
    print("Models Loaded Successfully!")
except Exception as e:
    print(f"Error loading models or feature columns: {e}")
    SVM_MODEL = None
    NB_MODEL = None


# --- 2. FUNGSI PRE-PROCESSING INPUT BARU ---
# Fungsi untuk membersihkan dan meng-encode input dari website
def preprocess_input(input_data):
    # Mengubah data input menjadi DataFrame agar sesuai dengan format model
    df_input = pd.DataFrame([input_data])
    
    # Melakukan ENCODING yang sama seperti Tahap 2 (Survey Data)
    
    # 1. Encoding Binary/Ordinal
    df_input['family_history'] = df_input['family_history'].map({'Yes': 1, 'No': 0})
    df_input['treatment'] = df_input['treatment'].map({'Yes': 1, 'No': 0})
    
    mapping_wi = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    df_input['work_interfere'] = df_input['work_interfere'].map(mapping_wi)

    # 2. Encoding Gender (Male=1 vs Female/Other=0)
    g = str(df_input['Gender'].iloc[0]).lower().strip()
    df_input['Gender'] = 1 if 'male' in g or g in ['m', 'male', 'cis male', 'm.'] else 0
    
    # Memastikan urutan kolom input sama persis dengan urutan FEATURE_COLUMNS
    return df_input[FEATURE_COLUMNS].values


# --- 3. API ENDPOINT PREDIKSI ---
@app.route('/predict', methods=['POST'])
def predict():
    if not SVM_MODEL or not NB_MODEL:
        return jsonify({'error': 'Models not loaded'}), 500
        
    data = request.get_json(force=True)
    
    # Panggil fungsi preprocessing
    try:
        processed_input = preprocess_input(data)
    except Exception as e:
        return jsonify({'error': f'Input preprocessing error: {e}'}), 400

    # Lakukan Prediksi SVM
    svm_pred_class = SVM_MODEL.predict(processed_input)[0]
    
    # Lakukan Prediksi Naive Bayes (dengan probabilitas)
    nb_pred_class = NB_MODEL.predict(processed_input)[0]
    nb_probs = NB_MODEL.predict_proba(processed_input)[0]
    
    # Mapping label numerik kembali ke teks
    class_mapping = {0: 'No Depression', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    
    # Format output probabilitas untuk Naive Bayes
    nb_prob_output = {class_mapping[i]: round(nb_probs[i] * 100, 2) for i in range(len(nb_probs))}

    return jsonify({
        'svm_diagnosis': class_mapping[svm_pred_class],
        'nb_diagnosis': class_mapping[nb_pred_class],
        'nb_probabilities': nb_prob_output
    })

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    # Pastikan server berjalan pada mode debug untuk pengembangan
    app.run(debug=True)