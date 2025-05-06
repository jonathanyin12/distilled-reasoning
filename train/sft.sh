uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false
dataset_file_path="jonathanyin/aime_1983_2023_qwq-32b_traces_16384"
output_dir="ckpts/qwen2.5-32b_qwq-32b_traces_16384_${uid}"
dataset_text_field="templated_response"
max_length=32768

# --- Enable Hugging Face Transfer ---
export HF_HUB_ENABLE_HF_TRANSFER=1

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    sft.py \
    --base_model=${base_model} \
    --dataset_file_path=${dataset_file_path} \
    --dataset_text_field=${dataset_text_field} \
    --max_length=${max_length} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${output_dir} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True 
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'