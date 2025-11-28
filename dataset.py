"""PD Scalogram Dataset Loader.

- Input: 64×64 RGB images
- Normalization: Auto-computed from dataset
- Split: 75 samples/class for val/test, remainder for training
"""
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


CLASS_MAP = {'corona': 0, 'no_pd': 1, 'surface': 2}


class PDScalogram:
    """Dataset loader with auto-computed normalization."""
    
    def __init__(self, data_path, eval_per_class=75):
        """
        Args:
            data_path: Path to dataset directory
            eval_per_class: Samples reserved for val/test per class (default: 75)
        """
        self.data_path = os.path.abspath(data_path)
        self.eval_per_class = eval_per_class
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        self._compute_stats()
        self._load_data()
        self._shuffle_all()
    
    def _compute_stats(self):
        """Compute per-channel mean and std."""
        print('Computing mean/std...')
        pixels = []
        
        for class_name in CLASS_MAP:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            for fname in os.listdir(class_path):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = Image.open(os.path.join(class_path, fname)).convert('RGB')
                pixels.append(self._base_transform(img).numpy())
        
        all_imgs = np.stack(pixels)  # (N, 3, H, W)
        self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
        self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_data(self):
        """Load and split data into train/val/test."""
        # Find min class size
        class_sizes = {}
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_sizes[class_name] = len(files)
        
        min_size = min(class_sizes.values())
        eval_size = min(self.eval_per_class, min_size)
        
        print(f'Split: {eval_size}/class for val/test, rest for train')
        
        for class_name, label in CLASS_MAP.items():
            path = os.path.join(self.data_path, class_name)
            if not os.path.exists(path):
                print(f'  Warning: {path} not found')
                continue
            
            files = sorted([f for f in os.listdir(path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            # Split: eval_size for val/test, rest for train
            eval_files = files[:eval_size]
            train_files = files[eval_size:]
            
            # Load images
            for fname in train_files:
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                self.X_train.append(self.transform(img).numpy())
                self.y_train.append(label)
            
            for fname in eval_files:
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                tensor = self.transform(img).numpy()
                self.X_val.append(tensor)
                self.y_val.append(label)
                self.X_test.append(tensor)
                self.y_test.append(label)
        
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
            idx = list(range(len(X)))
            random.Random(seed).shuffle(idx)
            X[:] = X[idx]
            y[:] = y[idx]
