# PD Scalogram Few-Shot Learning

Few-shot learning for Partial Discharge classification using scalogram images.

## Models
- **CovaMNet**: Covariance Metric Network
- **ProtoNet**: Prototypical Network  
- **CosineNet**: Cosine Similarity Network

## Project Structure
```
main.py          # Entry point
dataset.py       # Data loading (64x64, auto-normalized)
dataloader/      # Episodic sampling
net/             # Model architectures
function/        # Loss & utilities
checkpoints/     # Saved weights
results/         # Metrics & plots
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train
python main.py --model covamnet --shot_num 1 --mode train

# Test
python main.py --model covamnet --shot_num 1 --mode test
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | covamnet | cosine, protonet, covamnet |
| `--way_num` | 3 | Classes per episode |
| `--shot_num` | 1 | Support samples per class |
| `--query_num` | 19/15 | Query samples (1-shot/5-shot) |
| `--episode_num_train` | 100 | Episodes per epoch |
| `--episode_num_test` | 75 | Test episodes |
| `--lr` | 1e-3 | Learning rate |
| `--gamma` | 0.1 | LR decay factor |

## Dataset
- **Input**: 64×64 RGB images
- **Split**: 70% train / 30% test
- **Normalization**: Auto-computed from dataset
- **Classes**: corona, no_pd, surface
