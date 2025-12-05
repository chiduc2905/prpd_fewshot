"""
ADVANCED DATA LEAKAGE CHECKS
1. Train-Val contamination
2. Seed consistency
3. Episode statistics
"""

import torch
import numpy as np
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset

# Load dataset
dataset = PDScalogram('./prpd_images_for_cnn/', val_per_class=40, test_per_class=40)

print("=" * 80)
print("CHECK 1: TRAIN-VAL-TEST CONTAMINATION")
print("=" * 80)

# Convert to numpy for easier comparison
train_data = dataset.X_train
val_data = dataset.X_val
test_data = dataset.X_test

print(f"Train size: {len(train_data)}")
print(f"Val size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# Check for exact duplicates
train_val_overlap = 0
train_test_overlap = 0
val_test_overlap = 0

for i, train_img in enumerate(train_data):
    for j, val_img in enumerate(val_data):
        if np.array_equal(train_img, val_img):
            train_val_overlap += 1
            print(f"❌ Train[{i}] == Val[{j}]")
            
for i, train_img in enumerate(train_data):
    for j, test_img in enumerate(test_data):
        if np.array_equal(train_img, test_img):
            train_test_overlap += 1
            print(f"❌ Train[{i}] == Test[{j}]")

for i, val_img in enumerate(val_data):
    for j, test_img in enumerate(test_data):
        if np.array_equal(val_img, test_img):
            val_test_overlap += 1
            print(f"❌ Val[{i}] == Test[{j}]")

print(f"\nTrain-Val overlap: {train_val_overlap}")
print(f"Train-Test overlap: {train_test_overlap}")
print(f"Val-Test overlap: {val_test_overlap}")

if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
    print("🚨 CRITICAL: Found data contamination!")
else:
    print("✅ No contamination between splits")

print("\n" + "=" * 80)
print("CHECK 2: SEED CONSISTENCY")
print("=" * 80)

# Create val and test loaders with SAME settings
val_loader = FewshotDataset(
    data=torch.from_numpy(val_data).float(),
    labels=torch.from_numpy(dataset.y_val).long(),
    episode_num=5,
    way_num=2,
    shot_num=1,
    query_num=1,
    seed=1  # Val seed
)

test_loader = FewshotDataset(
    data=torch.from_numpy(test_data).float(),
    labels=torch.from_numpy(dataset.y_test).long(),
    episode_num=5,
    way_num=2,
    shot_num=1,
    query_num=1,
    seed=2  # Test seed - DIFFERENT from val!
)

print(f"Val loader seed: {val_loader.seed}")
print(f"Test loader seed: {test_loader.seed}")
print("⚠️  Different seeds → Different episode sampling strategies")
print("   This could cause Val Acc ≠ Test Acc even without leakage")

print("\n" + "=" * 80)
print("CHECK 3: CLASS DISTRIBUTION IN SPLITS")
print("=" * 80)

train_class_counts = np.bincount(dataset.y_train, minlength=2)
val_class_counts = np.bincount(dataset.y_val, minlength=2)
test_class_counts = np.bincount(dataset.y_test, minlength=2)

print(f"Train: Class 0={train_class_counts[0]}, Class 1={train_class_counts[1]}")
print(f"Val:   Class 0={val_class_counts[0]}, Class 1={val_class_counts[1]}")
print(f"Test:  Class 0={test_class_counts[0]}, Class 1={test_class_counts[1]}")

if val_class_counts[0] != val_class_counts[1]:
    print("⚠️  Val set is imbalanced!")
if test_class_counts[0] != test_class_counts[1]:
    print("⚠️  Test set is imbalanced!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Possible reasons for Val Acc > Test Acc:")
print("1. Different seeds → Different episode difficulty")
print("2. Small sample size (80 val, 80 test) → High variance")
print("3. Model overfitting to validation set from hyperparameter tuning")
print("4. Val and Test drawn from slightly different distributions")
print("=" * 80)
