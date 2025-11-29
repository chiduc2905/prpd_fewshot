#!/bin/bash
# Train all models with 1-shot for different training samples

SAMPLES=(30 60 90)
MODELS=("covamnet" "protonet" "cosine")

echo "=========================================="
echo "Training 1-shot models"
echo "=========================================="

for sample in "${SAMPLES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training with $sample samples (1-shot)"
    echo "=========================================="
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo ">>> $model 1-shot ($sample samples)"
        python3 main.py --model $model --shot_num 1 --training_samples $sample --mode train
    done
done

echo ""
echo "=========================================="
echo "All 1-shot training completed!"
echo "=========================================="
