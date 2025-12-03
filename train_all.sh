#!/bin/bash
# Train all models with both 1-shot and 5-shot for all training samples

SAMPLES=(12 60 "all")
MODELS=("covamnet" "protonet" "cosine")
SHOTS=(1 5)

echo "=============================================================="
echo "Training all models (1-shot and 5-shot) for all training sizes"
echo "=============================================================="

for shot in "${SHOTS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "Training all models with ${shot}-shot"
    echo "=============================================================="
    
    for sample in "${SAMPLES[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "Training with $sample samples (${shot}-shot)"
        echo "------------------------------------------"
        
        for model in "${MODELS[@]}"; do
            echo ""
            echo ">>> $model ${shot}-shot ($sample samples)"
            if [ "$sample" == "all" ]; then
                python3 main.py --model $model --shot_num $shot --mode train
            else
                python3 main.py --model $model --shot_num $shot --training_samples $sample --mode train
            fi
        done
    done
done

echo ""
echo "=============================================================="
echo "All training completed!"
echo "=============================================================="
echo "Results saved in results/ directory"
echo "Per-model summary files: covamnet.txt, protonet.txt, cosine.txt"
echo "=============================================================="
