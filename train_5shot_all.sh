#!/bin/bash
# Train all models with 5-shot for different training samples

SAMPLES=(12 60 "all")
MODELS=("covamnet" "protonet" "cosine")

echo "=========================================="
echo "Training 5-shot models"
echo "=========================================="

for sample in "${SAMPLES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training with $sample samples (5-shot)"
    echo "=========================================="
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo ">>> $model 5-shot ($sample samples)"
        if [ "$sample" == "all" ]; then
            python3 main.py --model $model --shot_num 5 --mode train
        else
            python3 main.py --model $model --shot_num 5 --training_samples $sample --mode train
        fi
    done
done

echo ""
echo "=========================================="
echo "All 5-shot training completed!"
echo "=========================================="
