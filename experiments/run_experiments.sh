#!/bin/bash

# Script to run multiple GraphToken experiments
# Make sure to activate your virtual environment before running

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of experiments directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# --- Configuration ---
# Add more values to these arrays to expand the experiments
TASKS=("CONNECTED" "MVC" "MIS")
DIFFICULTIES=("easy" "hard")
MODELS=("GIN" "GAT" "GCN")
HIDDEN_DIMS=(64)
NUM_LAYERS=(3)
LPE_DIMS=(4)

# You can now specify multiple epoch values to sweep
EPOCHS=(20)
# Add your HuggingFace token if required for the LLM
# HF_TOKEN="your-token-here"

# You might need to change this depending on your setup
LLM="google/gemma-3-4b-it"

# Quantization precision: 4bit, 8bit, 16bit, or 32bit
# Note: 4bit and 8bit require CUDA and bitsandbytes library
PRECISIONS=("4bit")

# --- Experiment Loop ---
for task in "${TASKS[@]}"; do
    for difficulty in "${DIFFICULTIES[@]}"; do
        for gnn_model in "${MODELS[@]}"; do
            for hidden_dim in "${HIDDEN_DIMS[@]}"; do
                for num_layers in "${NUM_LAYERS[@]}"; do
                    for lpe_dim in "${LPE_DIMS[@]}"; do
                        for epochs in "${EPOCHS[@]}"; do
                            for precision in "${PRECISIONS[@]}"; do
                        
                                echo "================================================================="
                                echo "Starting experiment:"
                                echo "  Task:         $task"
                                echo "  Difficulty:   $difficulty"
                                echo "  GNN Model:    $gnn_model"
                                echo "  Hidden Dim:   $hidden_dim"
                                echo "  Num Layers:   $num_layers"
                                echo "  LPE Dim:      $lpe_dim"
                                echo "  LLM:          $LLM"
                                echo "  Precision:    $precision"
                                echo "  Epochs:       $epochs"
                                echo "================================================================="
                                
                                # Change to project root directory
                                cd "$PROJECT_ROOT"
                                
                                CMD="uv run python train_gnn.py \
                                    --mode graphtoken \
                                    --task \"$task\" \
                                    --difficulty \"$difficulty\" \
                                    --model \"$gnn_model\" \
                                    --llm \"$LLM\" \
                                    --hidden-dim \"$hidden_dim\" \
                                    --num-layers \"$num_layers\" \
                                    --lpe-dim \"$lpe_dim\" \
                                    --precision \"$precision\" \
                                    --epochs \"$epochs\" \
                                    --save-model"
                                
                                # Add HF token if provided
                                if [ -n "$HF_TOKEN" ]; then
                                    CMD="$CMD --hf-token $HF_TOKEN"
                                fi
                                
                                # Execute command
                                eval "$CMD"
                                
                                echo -e "\n\n"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiments finished."
