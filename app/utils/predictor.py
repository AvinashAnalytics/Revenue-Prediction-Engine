import numpy as np

def preprocess_input(inputs, feature_scaler):
    """
    Preprocess user inputs for model prediction.
    
    Args:
        inputs (dict): Dictionary of user inputs
        feature_scaler: Fitted feature scaler
    
    Returns:
        np.array: Scaled and reshaped input array
    """
    category_map = {'Fast Food': 0, 'Casual Dining': 1, 'Fine Dining': 2}
    
    processed = [
        inputs['franchise'],
        category_map[inputs['category']],
        inputs['menu_size'],
        inputs['orders'],
        inputs['orders'] / inputs['menu_size']
    ]
    
    return feature_scaler.transform(np.array(processed).reshape(1, -1))

def inverse_scale_prediction(scaled_prediction, target_scaler):
    """
    Convert scaled predictions back to original scale.
    
    Args:
        scaled_prediction: Model's output
        target_scaler: Fitted target scaler
    
    Returns:
        float: Prediction in original scale
    """
    return target_scaler.inverse_transform(scaled_prediction)[0][0]