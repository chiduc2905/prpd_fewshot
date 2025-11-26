"""PD Scalogram Dataset Loader.

Input: 64x64 RGB images, auto-normalized from dataset statistics.
Split: Train (fixed by --training_samples) / Val (50% remaining) / Test (50% remaining).
"""
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random
import torchvision.transforms as transforms

class_idx = {'corona': 0, 'no_pd': 1, 'surface': 2}


class PDScalogram:
    """Dataset loader with auto-computed normalization."""
    
    def __init__(self, data_path, total_training_samples=None):
        """
        Args:
            data_path: Path to dataset directory.
            total_training_samples: Fixed total training samples (distributed evenly across classes).
                                    Remaining samples split 50/50 for val/test.
        """
        self.data_path = data_path
        self.total_training_samples = total_training_samples
        
        # Normalize path: handle relative and absolute paths
        if not os.path.isabs(data_path):
            self.data_path = os.path.abspath(data_path)
        
        self.classes = sorted(list(class_idx.keys()), key=lambda c: class_idx[c])
        self.nclasses = len(self.classes)
        
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []
        
        # Initial transform (without normalization) to compute mean/std
        self.resize_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Will be set after computing dataset statistics
        self.transform = None
        self.mean = None
        self.std = None

        print(f'Using dataset path: {self.data_path}')
        self._compute_mean_std()
        self._load_data()
        self._shuffle()
    
    def _compute_mean_std(self):
        """Compute per-channel mean and std from dataset."""
        print("Computing dataset mean and std...")
        all_pixels = []
        
        for class_name in class_idx.keys():
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for fname in files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.resize_transform(img)  # (3, H, W)
                all_pixels.append(img_tensor)
        
        # Stack all images: (N, 3, H, W)
        all_images = np.stack([t.numpy() for t in all_pixels], axis=0)
        
        # Compute mean and std per channel
        self.mean = all_images.mean(axis=(0, 2, 3)).tolist()  # [mean_R, mean_G, mean_B]
        self.std = all_images.std(axis=(0, 2, 3)).tolist()    # [std_R, std_G, std_B]
        
        print(f"Dataset mean: {self.mean}")
        print(f"Dataset std: {self.std}")
        
        # Define final transform with computed normalization
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def _load_data(self):
        """Load and split data: Train (fixed) / Val (50% remain) / Test (50% remain)."""
        
        # First pass: find min class size for balancing
        class_file_counts = {}
        for class_name in class_idx.keys():
            class_path = os.path.join(self.data_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_file_counts[class_name] = len(files)
        
        min_class_size = min(class_file_counts.values())
        print(f"Balancing dataset: using {min_class_size} samples per class (based on min class)")
        
        # Reserve a fixed eval pool per class (val & test use the same pool)
        eval_per_class = min(75, min_class_size)
        print(f"Reserving {eval_per_class} samples per class for validation/test pool. Remaining samples go to training pool.")
        
        for class_name, class_label in class_idx.items():
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path not found {class_path}")
                continue
                
            files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if len(files) == 0:
                print(f"Warning: Class {class_name} has no images.")
                continue
            
            # Shuffle files with fixed seed for reproducibility
            random.Random(42).shuffle(files)
            
            # Limit to min_class_size for balance
            files = files[:min_class_size]

            # Reserve eval_per_class files for val/test (both will use this pool)
            eval_files = files[:eval_per_class]
            remaining_files = files[eval_per_class:]
            # Remaining files are used as the training pool (all available)
            train_files = remaining_files
            val_files = eval_files
            test_files = eval_files
            
            # Load Train
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_train.append(img_tensor.numpy())
                self.y_train.append(class_label)
            
            # Load Val and Test from the same eval pool
            for fname in val_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_val.append(img_tensor.numpy())
                self.y_val.append(class_label)
            for fname in test_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_test.append(img_tensor.numpy())
                self.y_test.append(class_label)
                
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f"Data loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        
        # Distribution check
        for name, y in [('Train', self.y_train), ('Val', self.y_val), ('Test', self.y_test)]:
            unique, counts = np.unique(y, return_counts=True)
            print(f"{name} distribution: {dict(zip(unique, counts))}")

    def _shuffle(self):
        # Shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))

        # Shuffle val samples
        index = list(range(self.X_val.shape[0]))
        random.Random(0).shuffle(index)
        self.X_val = self.X_val[index]
        self.y_val = np.array(tuple(self.y_val[i] for i in index))

        # Shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))
