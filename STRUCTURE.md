# Model Training Commands

Use the following commands to train the different models. Ensure your dataset is located at `./scalogram_images/` or specify the path with `--dataset_path`.

## CovaMNet (Covariance Metric Network)

**1-Shot Training:**
```bash
python main.py --model covamnet --shot_num 1 --query_num 15 --num_epochs 100 --mode train
```

**5-Shot Training:**
```bash
python main.py --model covamnet --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

## ProtoNet (Prototypical Network)

**1-Shot Training:**
```bash
python main.py --model protonet --shot_num 1 --query_num 15 --num_epochs 100 --mode train
```

**5-Shot Training:**
```bash
python main.py --model protonet --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

## CosineNet (Cosine Similarity Network)

**1-Shot Training:**
```bash
python main.py --model cosine --shot_num 1 --query_num 15 --num_epochs 100 --mode train
```

**5-Shot Training:**
```bash
python main.py --model cosine --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

---

## Testing Models

To test a trained model, change `--mode train` to `--mode test`.

**Example:**
```bash
python main.py --model covamnet --shot_num 1 --mode test
```
