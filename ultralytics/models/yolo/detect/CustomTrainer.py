# File: ultralytics/models/yolo/detect/CustomTrainer.py

from collections import defaultdict
import random
import numpy as np

import torch.nn as nn

from ultralytics.data import build_balanced_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.class_balance_utils import update_model_class_weights

class ImbalancedDetectionTrainer(DetectionTrainer):
    """
    Extended DetectionTrainer for handling class imbalance problems.
    Includes data oversampling, class weighting, and enhanced augmentation.
    """
    
    # def __init__(self, class_counts, overrides=None, _callbacks=None):
    def __init__(self, overrides=None, _callbacks=None):
        """Initialize trainer with class counts for imbalance handling."""
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        
        
        
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset with enhanced augmentation for minority classes."""
        dataset = super().build_dataset(img_path, mode, batch)
        #--------------------
        class_counts = self.get_class_counts(dataset)
        self.class_counts = class_counts
        
        # Identify minority and majority classes
        counts = list(class_counts.values())
        median_count = np.median(counts)
        self.minority_classes = [cls for cls, count in class_counts.items() if count < median_count * 0.5]
        LOGGER.info(f"Identified minority classes: {self.minority_classes}")
        #--------------------
        if mode == "train" and self.minority_classes:
            # Apply stronger augmentation to minority classes
            dataset.minority_classes = self.minority_classes
            
            # Store original augment method
            original_augment = dataset._augment
            
            # Define enhanced augmentation for minority classes
            def enhanced_augment(img, labels):
                # Apply standard augmentation
                img, labels = original_augment(img, labels)
                
                # Check if image contains minority classes
                if any(cls in self.minority_classes for cls in labels["cls"]):
                    # Apply additional augmentation for minority classes
                    if random.random() < 0.5:  # Apply more aggressive transforms 50% of time
                        # Example: additional horizontal flips or rotations
                        if random.random() < 0.5:
                            img = np.fliplr(img)
                            labels["bboxes"][:, [0, 2]] = 1 - labels["bboxes"][:, [2, 0]]
                            
                return img, labels
                
            # Replace augment method with enhanced version
            dataset._augment = enhanced_augment
            
        return dataset
    
    def get_class_counts(self, dataset):
        class_counts = {
            "GRAFFITI": 950, 
            "FADED_SIGNAGE": 100, 
            "POTHOLES": 2100, 
            "GARBAGE": 7000, 
            "CONSTRUCTION_ROAD": 2300, 
            "BROKEN_SIGNAGE": 50, 
            "BAD_STREETLIGHT": 1250, 
            "BAD_BILLBOARD": 600, 
            "SAND_ON_ROAD": 1800, 
            "CLUTTER_SIDEWALK": 100
        }
        return class_counts
        # # ssss = dataset.get_labels()
        # class_counts = defaultdict(int)
        # for i in range(len(dataset)):
        #     # Assuming the dataset's labels are accessible and are in the format of class_id, x, y, w, h
        #     labels = dataset[i]["cls"]  # Adjust based on how your dataset stores labels
        #     # Iterate through the labels of the current image
        #     for label in labels:
        #         class_id = label[0]  # Class ID is usually the first element
        #         class_counts[class_id] += 1
        
        # return class_counts
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Get dataloader with class balancing for train mode."""
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # Use balanced sampler for training
        if mode == "train" and not getattr(dataset, "rect", False):
            return build_balanced_dataloader(
                dataset, 
                batch_size, 
                self.args.workers, 
                self.class_counts,
                shuffle=True, 
                rank=rank,
                num_replicas=getattr(self, 'world_size', 1) if rank != -1 else 1
            )
        
        # Use standard dataloader for validation
        return super().get_dataloader(dataset_path, batch_size, rank, mode)
    
    def _setup_train(self, *args, **kwargs):
        """Set up training and apply class weights to model."""
        result = super()._setup_train(*args, **kwargs)
        
        # Apply class weights to model
        if hasattr(self, 'model') and self.model is not None:
            update_model_class_weights(self.model, self.class_counts)
        
        return result