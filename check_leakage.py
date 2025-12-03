"""
DATA LEAKAGE DIAGNOSIS SCRIPT
Run this to verify Support/Query overlap
"""

import torch
import numpy as np
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset

# Load dataset
dataset = PDScalogram('./prpd_images_for_cnn/', val_per_class=40, test_per_class=40)

# Convert to torch tensors
val_data = torch.from_numpy(dataset.X_val).float()
val_labels = torch.from_numpy(dataset.y_val).long()

# Create episode generator
val_loader = FewshotDataset(
    data=val_data,
    labels=val_labels,
    episode_num=10,  # Check 10 episodes
    way_num=2,
    shot_num=1,
    query_num=1,
    seed=1
)

print("=" * 80)
print("CHECKING FOR SUPPORT-QUERY OVERLAP")
print("=" * 80)

overlap_count = 0
for ep_idx in range(10):
    query_imgs, query_labels, support_imgs, support_labels = val_loader[ep_idx]
    
    # Check if any query image exists in support set
    for q_idx in range(len(query_imgs)):
        q_img = query_imgs[q_idx]
        for s_idx in range(len(support_imgs)):
            s_img = support_imgs[s_idx]
            if torch.equal(q_img, s_img):
                overlap_count += 1
                print(f"❌ LEAKAGE FOUND in Episode {ep_idx}:")
                print(f"   Query[{q_idx}] (label={query_labels[q_idx]}) == Support[{s_idx}] (label={support_labels[s_idx]})")
                print(f"   Same image in both sets!")

print("=" * 80)
if overlap_count > 0:
    print(f"🚨 CRITICAL: Found {overlap_count} overlaps!")
    print("This explains Val Acc = 100% (model just memorizes support set)")
else:
    print("✅ No overlap detected")
print("=" * 80)
