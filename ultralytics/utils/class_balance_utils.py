# File: ultralytics/utils/class_balance_utils.py

import torch
from ultralytics.utils import LOGGER

def update_model_class_weights(model, class_counts):
    """
    Apply class weights inversely proportional to frequency for handling imbalanced data.
    To be called after model initialization.
    """
    # Calculate weights inversely proportional to sample counts
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Create weight array with same ordering as model.names
    weights = torch.ones(num_classes, device=model.device)
    for i, name in enumerate(model.names):
        if name in class_counts and class_counts[name] > 0:
            weights[i] = total_samples / (num_classes * class_counts[name])
    
    # Normalize weights to avoid changing overall loss magnitude
    weights = weights / weights.mean()
    
    # Set class weights in the model's loss function
    # Depending on YOLO version, adjust how weights are applied
    if hasattr(model, 'loss'):
        model.loss.class_weights = weights
    elif hasattr(model, 'model') and hasattr(model.model[-1], 'class_weights'):
        model.model[-1].class_weights = weights
    
    LOGGER.info(f"Applied class weights for imbalanced data: {weights.tolist()}")
    return weights