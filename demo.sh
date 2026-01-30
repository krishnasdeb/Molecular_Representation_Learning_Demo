#!/bin/bash

clear
echo "Starting Demo"

if [ ! -f "best_pretrain_model.pt" ]; then
    echo "No pretrained weights found. Running Pretraining first"
    python pretrain_model.py
else
    echo "Found existing pretrained weights. Skipping pretrain"
fi

echo "Starting Finetuning"
python finetune_model.py

echo "Demo Complete"