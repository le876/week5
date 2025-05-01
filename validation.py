import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import rosenbrock_function

def validate_model(surrogate_model, X_test, y_test):
    """
    Validate surrogate model on test data
    
    Parameters:
        surrogate_model: Trained surrogate model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        metrics: Dictionary containing validation metrics
    """
    # Get predictions
    y_pred = surrogate_model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"Validation Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Return metrics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics

def compute_true_labels(selected_samples, selected_indices, X_full, y_full):
    """
    Compute true labels for selected samples
    
    Parameters:
        selected_samples: Features of selected samples, shape (n_samples, n_features)
        selected_indices: Indices of selected samples
        X_full: Full feature dataset
        y_full: Full target dataset
        
    Returns:
        selected_labels: True labels of selected samples
    """
    # Get true labels for selected samples
    selected_labels = y_full[selected_indices]
    
    print(f"Computed true labels for {len(selected_indices)} samples")
    
    return selected_labels

def evaluate_final_model(surrogate_model, X_test, y_test):
    """
    Evaluate final model performance on test set
    
    Parameters:
        surrogate_model: Trained surrogate model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        test_rmse: RMSE on test set
    """
    # Evaluate model on test set
    test_metrics = validate_model(surrogate_model, X_test, y_test)
    
    return test_metrics['rmse'] 