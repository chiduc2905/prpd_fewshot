import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random
import torchvision.transforms as transforms

class_idx = {
    'corona': 0,
    'no_pd': 1,
    'surface': 2
}

class PDScalogram:
    def __init__(self, data_path, total_training_samples=None):
        """
        Args:
            data_path: Path to dataset
            total_training_samples: If set (e.g. 30, 60), this is the TOTAL number of training samples 
                                    across all classes. They will be distributed evenly among classes.
                                    If None, use all available training data.
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
        self.X_test = []
        self.y_test = []
        
        # Define Normalization Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f'Using dataset path: {self.data_path}')
        self._load_data()
        self._shuffle()
        
    def _load_data(self):
        # Fixed test set size: 75 samples per class
        samples_per_class_test = 75
        
        # Calculate training samples per class
        samples_per_class_train = None
        if self.total_training_samples is not None:
            samples_per_class_train = self.total_training_samples // self.nclasses
            print(f"Limiting training data: {samples_per_class_train} samples per class "
                  f"(Total: {self.total_training_samples})")
            
        for class_name, class_label in class_idx.items():
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path not found {class_path}")
                continue
                
            files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Ensure we have enough files
            total_needed = samples_per_class_test
            if samples_per_class_train is not None:
                total_needed += samples_per_class_train
                
            if len(files) < total_needed:
                 print(f"Warning: Class {class_name} has {len(files)} images, but need {total_needed}.")
            
            # For reproducibility, we should shuffle or sort. 
            # We sorted above. Now we can shuffle with a fixed seed if we want consistent splits,
            # or just take the first N. Let's assume the files are randomized enough or we just take sorted.
            # To be safe and consistent with "training samples" selection, let's shuffle files locally with seed 42 first.
            random.Random(42).shuffle(files)
            
            # Split Test / Train
            # Strategy: Take Test set first (fixed 75), then Train set
            test_files = files[:samples_per_class_test]
            remaining_files = files[samples_per_class_test:]
            
            if samples_per_class_train is None:
                train_files = remaining_files
            else:
                if len(remaining_files) >= samples_per_class_train:
                    train_files = remaining_files[:samples_per_class_train]
                else:
                     print(f"Warning: Class {class_name} only has {len(train_files)} training samples, fewer than requested {samples_per_class_train}.")
                     train_files = remaining_files

            # Load Train
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                # self.transform handles ToTensor and Normalize
                img_tensor = self.transform(img)
                self.X_train.append(img_tensor.numpy())
                self.y_train.append(class_label)
            
            # Load Test
            for fname in test_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_test.append(img_tensor.numpy())
                self.y_test.append(class_label)
                
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f"Data loaded: Train: {len(self.X_train)} (Total), Test: {len(self.X_test)} (Total)")
        
        # Balance Check
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Training Class Distribution: {dict(zip(unique, counts))}")

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))
