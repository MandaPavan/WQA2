import os
import joblib
import numpy as np

UNITS_OF_CONCENTRATION = "ppm"

def load_models(models_dir="models"):
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models folder not found: {models_dir}")

    models = {}
    for fn in os.listdir(models_dir):
        if fn.lower().endswith(".pkl"):
            name = fn[:-4]  # remove .pkl
            path = os.path.join(models_dir, fn)
            try:
                with open(path, "rb") as f:
                    models[name] = joblib.load(f)
            except Exception as e:
                print(f"Error loading {fn}: {e}")
                continue

    if not models:
        raise ValueError("No .pkl files found in models/ folder")

    return models

def predict_concentration(model, lab_std):
    """
    model predicts log1p(concentration) as you trained
    lab_std = [L*, a*, b*] standard CIELAB
    """
    pred_log = float(model.predict([lab_std])[0])
    return float(np.expm1(pred_log))