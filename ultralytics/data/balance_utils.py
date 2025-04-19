# File: ultralytics/data/balance_utils.py

import os
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.utils import PIN_MEMORY
# from ultralytics.data.loaders import InfiniteDataLoader
# from ultralytics.utils import PIN_MEMORY
# from ultralytics.data.dataloaders import seed_worker

class BalancedClassSampler(Sampler):
    """
    Sampler that ensures balanced representation of classes in each batch.
    Works with YOLOv11's distributed training setup.
    """
    def __init__(self, dataset, batch_size, rank=-1, num_replicas=1, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Index images by the classes they contain
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            for cls in np.unique(label['cls']):
                self.class_indices[int(cls)].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # Calculate total length
        self.num_samples = len(dataset) // num_replicas
        self.total_size = self.num_samples * num_replicas
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
        indices = []
        # Create balanced batches
        class_cycles = [iter(self.class_indices[cls]) for cls in self.classes]
        class_idxs = list(range(self.num_classes))
        
        # Keep sampling until we have enough indices
        while len(indices) < self.total_size:
            if self.shuffle:
                random.shuffle(class_idxs)
                
            for cls_idx in class_idxs:
                try:
                    # Try to get an index with this class
                    img_idx = next(class_cycles[cls_idx])
                    indices.append(img_idx)
                    
                    if len(indices) >= self.total_size:
                        break
                except StopIteration:
                    # If we've used all indices for this class, reset
                    class_cycles[cls_idx] = iter(
                        self.class_indices[self.classes[cls_idx]] if self.shuffle 
                        else sorted(self.class_indices[self.classes[cls_idx]])
                    )
        
        # Ensure total_size is exact
        indices = indices[:self.total_size]
        
        # Subsample for distributed training
        if self.num_replicas > 1:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

def build_balanced_dataloader(dataset, batch, workers, class_counts=None, shuffle=True, rank=-1, num_replicas=1):
    """Return a DataLoader with class balancing for handling imbalanced datasets."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    
    # Use BalancedClassSampler instead of DistributedSampler
    sampler = None if rank == -1 else BalancedClassSampler(
        dataset=dataset,
        batch_size=batch,
        rank=rank,
        num_replicas=num_replicas,
        shuffle=shuffle,
        seed=6148914691236517205 + rank
    )
    
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + rank)
    
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )