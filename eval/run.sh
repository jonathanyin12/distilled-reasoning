#!/bin/bash

# --- Configuration for eval.py ---
TASK="lighteval|aime24|0|0,lighteval|aime25|0|0"
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

# --- Job Tracking ---
declare -A PID_TO_GPU # Associative array: PID -> GPU_ID
PIDS=() # Array to store PIDs of active background jobs
JOB_INDEX=0

echo "Starting up to $MAX_JOBS evaluation jobs..."

while [[ $JOB_INDEX -lt $MAX_JOBS ]]; do
  # --- 1. Cleanup finished jobs ---
  # Check running PIDs and rebuild the list of active jobs and their GPU assignments
  CURRENT_PIDS=()
  declare -A CURRENT_PID_TO_GPU # Temporary map for active jobs in this iteration
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      # Job is still running
      CURRENT_PIDS+=("$pid")
      CURRENT_PID_TO_GPU[$pid]=${PID_TO_GPU[$pid]} # Keep track of its GPU
    else
      # Job finished
      FINISHED_GPU=${PID_TO_GPU[$pid]}
      echo "Job with PID $pid on GPU $FINISHED_GPU finished."
      # The GPU is implicitly freed as this PID won't be in the active list below
    fi
  done
  PIDS=("${CURRENT_PIDS[@]}") # Update PIDS to only contain active ones
  # Rebuild PID_TO_GPU from the active jobs found
  unset PID_TO_GPU
  declare -A PID_TO_GPU
  for pid in "${PIDS[@]}"; do
      PID_TO_GPU[$pid]=${CURRENT_PID_TO_GPU[$pid]}
  done

  # --- 2. Find a free GPU ---
  TARGET_GPU=""
  # Only look for a GPU if we haven't reached the max number of jobs AND we have capacity
  if [[ ${#PIDS[@]} -lt $NUM_GPUS ]]; then
      for gpu_id in "${ALLOCATED_GPUS[@]}"; do
          IS_BUSY=0
          # Check if this gpu_id is currently assigned to any *running* PID
          for assigned_gpu in "${PID_TO_GPU[@]}"; do # Iterate through values (assigned GPUs)
              if [[ "$assigned_gpu" == "$gpu_id" ]]; then
                  IS_BUSY=1
                  break # This GPU is busy
              fi
          done

          if [[ $IS_BUSY -eq 0 ]]; then
              # Found a free GPU
              TARGET_GPU=$gpu_id
              break # Use the first free GPU found
          fi
      done
  fi

  # --- 3. Launch or Wait ---
  if [[ -n "$TARGET_GPU" ]]; then
    # Found a free GPU ($TARGET_GPU), launch the next job
    CURRENT_JOB_NUM=$((JOB_INDEX + 1))
    CURRENT_SEED=$((SEED + CURRENT_JOB_NUM))
    echo "Launching job $CURRENT_JOB_NUM/$MAX_JOBS on available GPU $TARGET_GPU"

    # Set the base command and arguments using variables from the top
    COMMAND="python eval.py \
        --task ${TASK} \
        --model ${MODEL_NAME} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --seed ${CURRENT_SEED} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_model_length ${MAX_MODEL_LENGTH} \
        ${USE_CHAT_TEMPLATE}"

    # Run the command in the background, ensuring it only sees the assigned GPU
    echo "Running command: CUDA_VISIBLE_DEVICES=$TARGET_GPU $COMMAND > /dev/null 2>&1"
    # Ensure environment variables potentially affecting caching are handled if needed
    CUDA_VISIBLE_DEVICES=$TARGET_GPU $COMMAND > /dev/null 2>&1 &
    PID=$!
    PIDS+=("$PID") # Add new PID to our list
    PID_TO_GPU[$PID]=$TARGET_GPU # Store PID -> GPU mapping
    echo "  Job $CURRENT_JOB_NUM started with PID $PID on GPU $TARGET_GPU"

    ((JOB_INDEX++)) # Increment the counter for total jobs launched

  else
    # No free GPU found OR we have already launched MAX_JOBS
    # Only need to wait if there are still jobs running AND we haven't launched all required jobs yet
    if [[ ${#PIDS[@]} -gt 0 && $JOB_INDEX -lt $MAX_JOBS ]]; then
        echo "All $NUM_GPUS GPUs seem busy. Waiting for a job to complete..."
        wait -n # Wait for *any* running background job (from this script) to finish
        # The loop will continue, and the cleanup section at the top will handle the finished job
        sleep 5 # Add a small delay to allow OS/GPU driver cleanup before next check
    elif [[ ${#PIDS[@]} -eq 0 && $JOB_INDEX -ge $MAX_JOBS ]]; then
        # All jobs launched and all jobs finished, exit loop cleanly
        echo "All launched jobs completed."
        break
    else
        # Either all jobs launched and some still running (wait outside loop handles this)
        # or some other state. A small sleep prevents tight loop potentially.
        sleep 5
    fi
  fi
done

# Wait for all remaining background jobs launched by this script
echo "All $MAX_JOBS jobs have been launched. Waiting for any remaining jobs to finish..."
wait # Waits for all background jobs started in this shell
echo "All jobs completed."
