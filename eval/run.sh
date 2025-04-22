#!/bin/bash

# --- Configuration for eval.py ---
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
TEMPERATURE=0.6
TOP_P=0.95
MAX_NEW_TOKENS=32768
MAX_MODEL_LENGTH=32768
USE_CHAT_TEMPLATE="--use_chat_template" # Set to "" to disable
OUTPUT_DIR=$MODEL_NAME
SEED=0

# --- Script Parameters ---
MAX_JOBS=16

# --- Slurm GPU Detection ---
# Get the list of assigned GPUs from Slurm (or environment)
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "Error: CUDA_VISIBLE_DEVICES is not set. Cannot determine allocated GPUs."
  echo "Ensure you are running this script within a Slurm allocation that includes GPUs,"
  echo "or set CUDA_VISIBLE_DEVICES manually (e.g., CUDA_VISIBLE_DEVICES=0,1)."
  exit 1
fi

IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#ALLOCATED_GPUS[@]}

if [[ $NUM_GPUS -eq 0 ]]; then
  echo "Error: No GPUs found in CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES)."
  exit 1
fi

echo "Detected $NUM_GPUS allocated GPUs: ${ALLOCATED_GPUS[*]}"

PIDS=() # Array to store PIDs of background jobs
JOB_INDEX=0

echo "Starting $MAX_JOBS evaluation jobs..."

while [[ $JOB_INDEX -lt $MAX_JOBS ]]; do
  # Clean up PIDS array: remove PIDs of finished jobs
  CURRENT_PIDS=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      CURRENT_PIDS+=("$pid")
    # else
      # Optionally echo here if a job finished, e.g.: echo "Job with PID $pid finished."
    fi
  done
  PIDS=("${CURRENT_PIDS[@]}")

  # Check if we have a free GPU slot
  if [[ ${#PIDS[@]} -lt $NUM_GPUS ]]; then
    # Calculate which GPU to use (round-robin)
    GPU_ARRAY_INDEX=$(( JOB_INDEX % NUM_GPUS ))
    GPU_ID=${ALLOCATED_GPUS[$GPU_ARRAY_INDEX]}

    CURRENT_JOB_NUM=$((JOB_INDEX + 1))
    echo "Launching job $CURRENT_JOB_NUM/$MAX_JOBS on GPU $GPU_ID (Device Index $GPU_ARRAY_INDEX)"

    # Set the base command and arguments using variables from the top
    CURRENT_SEED=$((SEED + CURRENT_JOB_NUM))
    COMMAND="python eval.py \
        --model ${MODEL_NAME} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --seed ${CURRENT_SEED} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_model_length ${MAX_MODEL_LENGTH} \
        ${USE_CHAT_TEMPLATE}"

    # Run the command in the background, ensuring it only sees the assigned GPU
    echo "Running command: CUDA_VISIBLE_DEVICES=$GPU_ID $COMMAND"
    CUDA_VISIBLE_DEVICES=$GPU_ID $COMMAND &
    PID=$!
    PIDS+=("$PID")
    echo "  Job $CURRENT_JOB_NUM started with PID $PID"

    ((JOB_INDEX++))
  else
    # All GPUs are busy, wait for the *next* background job to finish
    echo "All $NUM_GPUS GPUs are busy. Waiting for a job to complete..."
    wait -n
    # Loop will continue, PIDS will be cleaned, and a free slot should be found
  fi
done

# Wait for all remaining background jobs launched by this script
echo "All $MAX_JOBS jobs have been launched. Waiting for remaining jobs to finish..."
wait
echo "All jobs completed."
