"""PD Scalogram Dataset Loader.

- Input: 64×64 RGB images
- Normalization: Auto-computed from dataset
- Split: 30 samples/class for val, 30 samples/class for test, remainder for training
"""
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


CLASS_MAP = {'surface': 0, 'corona': 1}


class PDScalogram:
    """Dataset loader with auto-computed normalization (from training set only)."""
    
    def __init__(self, data_path, val_per_class=30, test_per_class=30):
        """
        Args:
            data_path: Path to dataset directory
            val_per_class: Samples reserved for validation per class
            test_per_class: Samples reserved for test per class
        """
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # Base transform (for computing stats)
        self._base_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        
        # 1. Identify splits (filenames only)
        train_files, val_files, test_files = self._split_files()
        
        # 2. Compute stats on TRAIN files only
        self._compute_stats(train_files)
        
        # 3. Load all data using computed stats
        self._load_images(train_files, val_files, test_files)
        
        # 4. Shuffle
        self._shuffle_all()
    
    def _split_files(self):
        """Identify files for each split."""
        train_files = {}  # {class: [files]}
        val_files = {}
        test_files = {}
        
        # Find min class size to balance
        class_sizes = {}
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_sizes[class_name] = len(files)
            else:
                class_sizes[class_name] = 0
        
        if not class_sizes:
            return {}, {}, {}

        min_size = min(class_sizes.values())
        if min_size == 0:
            print("Warning: Found empty class or no images.")
            return {}, {}, {}

        val_size = min(self.val_per_class, min_size)
        test_size = min(self.test_per_class, min_size - val_size)
        eval_total = val_size + test_size
        
        print(f'Split: {val_size}/class for val, {test_size}/class for test, rest for train')
        
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if not os.path.exists(path):
                continue
                
            files = sorted([f for f in os.listdir(path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            val_files[class_name] = files[:val_size]
            test_files[class_name] = files[val_size:eval_total]
            train_files[class_name] = files[eval_total:]
            
        return train_files, val_files, test_files

    def _compute_stats(self, train_files):
        """Compute per-channel mean and std from training files."""
        print('Computing mean/std from training set...')
        pixels = []
        
        for class_name, files in train_files.items():
            class_path = os.path.join(self.data_path, class_name)
            for fname in files:
                img = Image.open(os.path.join(class_path, fname)).convert('RGB')
                pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Standard transform (Val/Test)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # Train transform (Augmentation for PRPD)
        # - ColorJitter: Simulates sensor sensitivity/noise
        # - RandomErasing: Simulates missing data/occlusion (robustness)
        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0), 
        ])
    
    def _load_images(self, train_files, val_files, test_files):
        """Load images applying normalization."""
        
        def load_list(file_dict, dest_X, dest_y, transform):
            for class_name, files in file_dict.items():
                label = CLASS_MAP[class_name]
                path = os.path.join(self.data_path, class_name)
                for fname in files:
                    img = Image.open(os.path.join(path, fname)).convert('RGB')
                    dest_X.append(transform(img).numpy())
                    dest_y.append(label)
        
        load_list(train_files, self.X_train, self.y_train, self.train_transform)
        load_list(val_files, self.X_val, self.y_val, self.transform)
        load_list(test_files, self.X_test, self.y_test, self.transform)
        
        # Convert to arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        for X, y, seed in [(self.X_train, self.y_train, 0),
                           (self.X_val, self.y_val, 1),
                           (self.X_test, self.y_test, 2)]:
            if len(X) > 0:
                idx = list(range(len(X)))
                random.Random(seed).shuffle(idx)
                X[:] = X[idx]
                y[:] = y[idx]
