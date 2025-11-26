# Project Structure

## Architecture Overview

```
Input (3×64×64) → Conv64F_Encoder → Features (64×16×16) → Model Head → Scores
```

### Encoder: Conv64F_Encoder
4-layer CNN with 2 max-pooling layers. Output: 64×16×16 feature maps.

### Models
| Model | Head | Metric |
|-------|------|--------|
| CosineNet | AvgPool → FC(64) | Cosine similarity |
| ProtoNet | AvgPool | Negative Euclidean distance |
| CovaMNet | CovaBlock → Conv1d | Covariance-based similarity |

## File Structure

```
├── main.py              # Training & testing entry point
├── dataset.py           # PDScalogram loader (64×64, auto-norm)
├── dataloader/
│   └── dataloader.py    # FewshotDataset (episodic sampling)
├── net/
│   ├── encoder.py       # Conv64F_Encoder backbone
│   ├── cosine.py        # CosineNet
│   ├── protonet.py      # ProtoNet
│   ├── covamnet.py      # CovaMNet
│   ├── cova_block.py    # Covariance block
│   └── utils.py         # Weight initialization
├── function/
│   └── function.py      # Loss, metrics, visualization
├── checkpoints/         # Model weights
└── results/             # Metrics, plots
```

## Training Commands

```bash
# CovaMNet 1-shot
python main.py --model covamnet --shot_num 1 --mode train

# ProtoNet 5-shot
python main.py --model protonet --shot_num 5 --mode train

# CosineNet with limited samples
python main.py --model cosine --training_samples 60 --mode train
```

## Testing Commands

```bash
# Auto-load best checkpoint
python main.py --model covamnet --shot_num 1 --mode test

# Custom weights
python main.py --model covamnet --weights path/to/model.pth --mode test
```
