# 1. Synthetic Data
python newterm/4_generate_lora_data.py \
    --input_file benchmark_2023/new_terms.jsonl \
    --output_file benchmark_2023/train.json \
    --num_sentences 10 \
    --model LLMs/Llama-3-70B-Instruct

# 2. Training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
train_path=newterm/4_run_clm_lora.py

# Train Qwen
model_path=LLMs/Qwen-7B-Chat
model_save=LLMs/Qwen-7B-ft
torchrun --nnodes 1 --nproc_per_node 2 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed newterm/script/deepspeed_config_zero2.json \
    --model_name_or_path ${model_path} \
    --train_file benchmark_2023/train.json \
    --use_lora True \
    --train_steps 200 \
    --lora_config newterm/script/qwen_lora_config.json \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 20 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --bf16=True \
    --bf16_full_eval True \
    --ddp_timeout 3600 \
    --gradient_checkpointing True \
    --output_dir ${model_save}

# Train Llama
model_path=LLMs/Llama-2-7b-chat-hf
model_save=LLMs/Llama-2-7b-ft
torchrun --nnodes 1 --nproc_per_node 2 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed newterm/script/deepspeed_config_zero2.json \
    --model_name_or_path ${model_path} \
    --train_file benchmark_2023/train.json\
    --use_lora True \
    --train_steps 200\
    --lora_config newterm/script/llama_lora_config.json \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 20 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --bf16=True \
    --bf16_full_eval True \
    --ddp_timeout 3600 \
    --gradient_checkpointing True \
    --output_dir ${model_save}

# 3.Evaluation
python -u newterm/evaluation.py --lora_weights LLMs/Qwen-7B-ft/adapter_model  --model LLMs/Qwen-7B-Chat --year 2023
python -u newterm/evaluation.py --lora_weights LLMs/Llama-2-7b-ft/adapter_model --model LLMs/Llama-2-7b-chat-hf --year 2023