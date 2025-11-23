import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random

class_idx = {
    'corona': 0,
    'no_pd': 1,
    'surface': 2
}

class PDScalogram:
    def __init__(self, data_path, samples_per_class=None):
        """
        Args:
            data_path: Path to dataset
            samples_per_class: If set (e.g. 30, 60), strictly limit training samples per class.
                               If None, use all available training data.
        """
        self.data_path = data_path
        self.samples_per_class = samples_per_class
        
        # Normalize path: handle relative and absolute paths
        if not os.path.isabs(data_path):
            self.data_path = os.path.abspath(data_path)
        
        self.classes = sorted(list(class_idx.keys()), key=lambda c: class_idx[c])
        self.nclasses = len(self.classes)
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        print(f'Using dataset path: {self.data_path}')
        self._load_data()
        
    def _load_data(self):
        # Fixed test set size: 75 samples per class
        TEST_SAMPLES_PER_CLASS = 75
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                print(f'Warning: Class path not found: {class_path}')
                continue
                
            class_label = class_idx[class_name]
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            # filter labeled files
            image_files = [f for f in image_files if 'labeled' not in f]
            
            # Shuffle before split
            random.Random(42).shuffle(image_files)
            
            if len(image_files) < TEST_SAMPLES_PER_CLASS:
                print(f"Warning: Class {class_name} has fewer than {TEST_SAMPLES_PER_CLASS} images.")
                test_files = image_files
                train_files = []
            else:
                test_files = image_files[:TEST_SAMPLES_PER_CLASS]
                train_files = image_files[TEST_SAMPLES_PER_CLASS:]
            
            # Limit training samples if requested (e.g. 30, 60)
            if self.samples_per_class is not None:
                if len(train_files) > self.samples_per_class:
                    train_files = train_files[:self.samples_per_class]
                else:
                     print(f"Warning: Requested {self.samples_per_class} training samples for {class_name}, but only {len(train_files)} available.")

            # Load Train
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_array = np.array(img) / 255.0
                self.X_train.append(img_array)
                self.y_train.append(class_label)
            
            # Load Test
            for fname in test_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_array = np.array(img) / 255.0
                self.X_test.append(img_array)
                self.y_test.append(class_label)
                
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f"Data loaded: Train: {len(self.X_train)} (Total), Test: {len(self.X_test)} (Total)")
        
        # Balance Check (Optional, but good for logging)
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Training Class Distribution: {dict(zip(unique, counts))}")

    # No global shuffle here to preserve class grouping logic if needed, 
    # but main.py converts to Dataset/DataLoader which will shuffle.
