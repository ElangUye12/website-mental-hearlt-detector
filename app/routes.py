# app/routes.py
from flask import Blueprint, request, jsonify, render_template
from app import SVM_MODEL, NB_MODEL
from app.utils import preprocess_input
import numpy as np 

bp = Blueprint('main', __name__)

# Mapping label: sesuai dengan pelatihan model Anda
CLASS_MAPPING = {0: 'No Depression', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

# --- RUTE YANG HILANG: Untuk Halaman Utama ---
@bp.route('/')
def index():
    return render_template('index.html')

# --- RUTE YANG HILANG: Untuk Halaman Deteksi ---
@bp.route('/detect')
def detect():
    return render_template('detect.html')

# --- RUTE PREDIKSI (Logika Final Anda) ---
@bp.route('/predict', methods=['POST'])
def predict():
    if SVM_MODEL is None or NB_MODEL is None:
        return jsonify({'error': 'Models not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        processed_input = preprocess_input(data)

        # Inisialisasi hasil untuk kedua model
        results = {}

        # --- 1. Naive Bayes (NB) ---
        nb_probs = NB_MODEL.predict_proba(processed_input)[0]
        nb_confidence = round(np.max(nb_probs) * 100, 1)
        nb_pred_index = np.argmax(nb_probs) # Gunakan argmax untuk konsistensi
        
        results['nb_diagnosis'] = CLASS_MAPPING[nb_pred_index]
        results['nb_confidence'] = nb_confidence
        results['nb_probabilities'] = {
            CLASS_MAPPING[i]: round(float(nb_probs[i]) * 100, 1)
            for i in range(len(nb_probs))
        }

        # --- 2. SVM ---
        try:
            svm_probs = SVM_MODEL.predict_proba(processed_input)[0]
            svm_confidence = round(np.max(svm_probs) * 100, 1)
            svm_pred_index = np.argmax(svm_probs) # Gunakan argmax untuk konsistensi
            
            results['svm_diagnosis'] = CLASS_MAPPING[svm_pred_index]
            results['svm_confidence'] = svm_confidence
            results['svm_probabilities'] = {
                CLASS_MAPPING[i]: round(float(svm_probs[i]) * 100, 1)
                for i in range(len(svm_probs))
            }
        except AttributeError:
            # Jika predict_proba gagal (AttributeError), berikan default 0.0
            results['svm_diagnosis'] = CLASS_MAPPING[SVM_MODEL.predict(processed_input)[0]]
            results['svm_confidence'] = 0.0
            results['svm_probabilities'] = {
                 'No Depression': 0.0, 'Mild': 0.0, 'Moderate': 0.0, 'Severe': 0.0 
            }
        
        return jsonify(results)

    except Exception as e:
        print(f"Processing/Prediction Error: {str(e)}")
        return jsonify({'error': f'Processing/Prediction failed: {str(e)}'}), 400