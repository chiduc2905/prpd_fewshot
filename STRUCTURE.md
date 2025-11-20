# Project Structure Summary

Your PD Scalogram Fewshot Learning project has been set up with the following structure:

## Folder Organization

```
PRPD/
├── dataset.py                 # Data loader for scalogram images
├── train_1shot.py            # 1-shot training script
├── train_5shot.py            # 5-shot training script
├── test_1shot.py             # 1-shot testing script
├── test_5shot.py             # 5-shot testing script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── train_1shot.sh            # Shell script for 1-shot training
├── train_5shot.sh            # Shell script for 5-shot training
├── dataloader/
│   ├── __init__.py
│   └── dataloader.py         # FewshotDataset implementation
├── function/
│   ├── __init__.py
│   └── function.py           # Utilities: loss, metrics, seed
├── net/
│   ├── __init__.py
│   └── pam_mamba.py          # CovarianceNet model
└── checkpoints/              # Saved models directory
```

## Key Files

### dataset.py
- **PDScalogram class**: Loads images from 3 folders (corona, no_pd, surface)
- Filters out 'labeled' files automatically
- 80/20 train-test split
- Returns normalized numpy arrays (N, H, W, 3)

### dataloader/dataloader.py
- **FewshotDataset**: Generates fewshot episodes with support and query sets
- way_num=3 (Corona, NoPD, Surface)
- Configurable shot_num (1 or 5 samples per class)

### function/function.py
- **seed_func()**: Sets reproducible random seeds
- **ContrastiveLoss()**: Discriminative loss for fewshot learning
- **cal_accuracy_fewshot_1shot/5shot()**: Accuracy computation
- **predicted_fewshot_1shot/5shot()**: Prediction for evaluation

### train_1shot.py & train_5shot.py
- Loads PDScalogram dataset
- Creates fewshot episodes
- Trains CovarianceNet model
- Saves best model based on test accuracy

### test_1shot.py & test_5shot.py
- Loads trained model
- Tests on new fewshot episodes
- Prints accuracy, confusion matrix, classification report

### net/pam_mamba.py
- Placeholder CovarianceNet architecture
- Replace with actual PAM-Mamba network as needed

## Usage

### Training 1-shot
```bash
python train_1shot.py --dataset_path ../ML/scalogram_images/ --num_epochs 100
```

### Training 5-shot
```bash
python train_5shot.py --dataset_path ../ML/scalogram_images/ --num_epochs 100
```

### Testing
```bash
python test_1shot.py --dataset_path ../ML/scalogram_images/ --model_path checkpoints/best_model.pth
```

## Key Differences from Mamba-Bearing

1. **Dataset**: Adapted for image data (scalograms) instead of time-series
2. **Classes**: 3 PD classes instead of bearing fault classes
3. **Image shape**: (N, H, W, 3) RGB normalized images
4. **Data loading**: PIL for image loading instead of scipy.io.loadmat
5. **File filtering**: Excludes 'labeled' files automatically
6. **Simplified**: Removed spectrum conversion (already have images)

## Requirements

All dependencies in requirements.txt:
- torch >= 2.0.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0
- Pillow >= 8.0.0

Install with: `pip install -r requirements.txt`
