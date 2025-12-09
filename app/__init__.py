# app/__init__.py
from flask import Flask
from joblib import load
import os

# Global model variables
SVM_MODEL = None
NB_MODEL = None

def create_app():
    global SVM_MODEL, NB_MODEL
    app = Flask(__name__)

    # Load models
    try:
        model_dir = os.path.join(app.root_path, '..', 'models')
        SVM_MODEL = load(os.path.join(model_dir, 'svm_model.pkl'))
        NB_MODEL = load(os.path.join(model_dir, 'naive_bayes_model.pkl'))
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        SVM_MODEL = None
        NB_MODEL = None

    # Register blueprint
    from app.routes import bp
    app.register_blueprint(bp)

    return app