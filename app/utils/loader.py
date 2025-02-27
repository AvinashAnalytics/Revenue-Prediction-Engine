import joblib
import pandas as pd
from tensorflow.keras.models import load_model

def load_feature_scaler(path='models/feature_scaler.pkl'):
    """Load the feature scaler from disk."""
    return joblib.load(path)

def load_target_scaler(path='models/target_scaler.pkl'):
    """Load the target scaler from disk."""
    return joblib.load(path)

def load_keras_model(path='models/neural_network_model.keras'):
    """Load the trained Keras model."""
    return load_model(path)

def load_sample_data():
    """Load sample data for visualization."""
    return pd.DataFrame({
        'Category': ['Fast Food', 'Casual Dining', 'Fine Dining'],
        'Avg_Revenue': [250000, 450000, 750000]
    })