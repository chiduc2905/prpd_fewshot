# PD Scalogram Few-Shot Learning

Few-shot classification of Partial Discharge patterns using scalogram images.

## Models

| Model | Description |
|-------|-------------|
| **CovaMNet** | Covariance Metric Network |
| **ProtoNet** | Prototypical Network |
| **CosineNet** | Cosine Similarity Network |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (1-shot)
python main.py --model covamnet --shot_num 1 --mode train

# Train with limited samples
python main.py --model protonet --shot_num 5 --training_samples 60 --mode train

# Test
python main.py --model covamnet --shot_num 1 --mode test
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | covamnet | Model: cosine, protonet, covamnet |
| `--way_num` | 3 | Classes per episode |
| `--shot_num` | 1 | Support samples per class |
| `--query_num` | 15 | Query samples per class (training) |
| `--training_samples` | all | Total training samples (e.g., 30=10/class) |
| `--num_epochs` | 100/70 | Training epochs (1-shot/5-shot) |
| `--lr` | 1e-3 | Learning rate |

## Evaluation Protocol

### Training Phase
- Episodes: 100/epoch
- Support: K-shot per class
- Query: 15 per class
- Validation: 75 episodes, 1 query/class

### Final Test Phase
- **150 episodes**
- **1-shot, 1-query per class**
- Total predictions: 450 (150 × 3 classes)
- Confusion matrix: each row sums to 150

## Dataset

```
scalogram_images/
├── corona/    # Class 0
├── no_pd/     # Class 1  
└── surface/   # Class 2
```

- **Input**: 64×64 RGB
- **Split**: 75/class for val/test, rest for train
- **Normalization**: Auto-computed from dataset

## Results

Results saved to `results/`:
- `summary_*.txt` - Metrics table
- `confusion_matrix_*.png` - Confusion matrix
- `tsne_*.png` - t-SNE visualization
