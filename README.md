# PD Scalogram Fewshot Learning

Few-shot learning on Partial Discharge (PD) Scalogram images with 3 classes: Corona, No PD, and Surface.

## Dataset Structure

```
scalogram_images/
├── corona/              # Corona PD class
├── no_pd/              # No PD class
└── surface/            # Surface PD class
```

Images are automatically split into 80% training and 20% test sets. Files containing 'labeled' in the name are excluded.

## Installation

```bash
pip install -r requirements.txt
```

## Training

### 1-shot Learning
```bash
python train_1shot.py --dataset_path ../ML/scalogram_images/ --num_epochs 100
```

### 5-shot Learning
```bash
python train_5shot.py --dataset_path ../ML/scalogram_images/ --num_epochs 100
```

## Testing

### 1-shot Testing
```bash
python test_1shot.py --dataset_path ../ML/scalogram_images/ --model_path checkpoints/best_model.pth
```

### 5-shot Testing
```bash
python test_5shot.py --dataset_path ../ML/scalogram_images/ --model_path checkpoints/best_model.pth
```

## Configuration

All parameters can be set via command-line arguments:

- `--dataset_path`: Path to scalogram dataset
- `--model_name`: Model name prefix
- `--num_epochs`: Number of training epochs
- `--way_num`: Number of classes (default: 3)
- `--shot_num`: Number of samples per class
- `--lr`: Learning rate (default: 1e-3)
- `--device`: cuda or cpu
- `--batch_size`: Batch size (default: 1)
- `--episode_num_train`: Training episodes (default: 100)
- `--episode_num_test`: Testing episodes (default: 75)

## Output

Trained models are saved to `checkpoints/` directory with accuracy in filename:
- `pd_scalogram_1shot_0.9234.pth`
- `pd_scalogram_5shot_0.9567.pth`

## Architecture

- **Dataloader**: Fewshot episode generation with support and query sets
- **Model**: CovarianceNet with feature extraction and similarity matching
- **Loss**: ContrastiveLoss for discriminative learning
