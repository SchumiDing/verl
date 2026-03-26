#!/bin/bash

set -e

# 1. 设置Ray和其他环境变量
ulimit -n 65535
export RAY_DEDUP_LOGS=0

# 单节点 Ray 配置（8卡）
RAY_PORT=${RAY_PORT:-6379}
NODE_RANK=${NODE_RANK:-0}
NUM_CPUS_PER_NODE=${NUM_CPUS_PER_NODE:-96}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

# 【重要】输出目录与 VERL_FILE_LOGGER_PATH 必须在 ray start 之前 export
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/shared-storage-user/mineru4s/dingruiyi/wanjuan-0314/checkpoints_rl/multiroute_distill_grpo_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"
export VERL_FILE_LOGGER_PATH="$OUTPUT_DIR/log.jsonl"

# 切换到工作目录
cd /mnt/shared-storage-user/mineru4s/dingruiyi/verl_wanjuan

# 2. 定义模型和数据路径
ACTOR_MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/WanJRxn_Downstream/output/multiroute_sft"
TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/wanjuan-0314/multiroute_train_distill.parquet"
VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/wanjuan-0314/multiroute_train_distill.parquet"

# 3. 训练参数
NNODES=1
TRAIN_BATCH_SIZE=256
LEARNING_RATE=1e-5
TOTAL_EPOCHS=10
ROLLOUT_N=8

export TIKTOKEN_ENCODINGS_BASE=/root/encoder
export TIKTOKEN_RS_CACHE_DIR=/root/encoder

echo "=========================================="
echo "Training Configuration:"
echo "  Model: $ACTOR_MODEL_PATH"
echo "  Train Data: $TRAIN_DATA"
echo "  Val Data: $VAL_DATA"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Batch Size: $TRAIN_BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Total Epochs: $TOTAL_EPOCHS"
echo "  Rollout N: $ROLLOUT_N"
echo "  GPUs per Node: $NUM_GPUS_PER_NODE"
echo "  Nodes: $NNODES"
echo "  Reward Manager: multiroute_distill_cot"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=multiroute_distill_cot \
    reward.reward_manager.source=register \
    reward.num_workers=8 \
    reward.reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "file"]' \
    trainer.project_name='multiroute_distill_grpo' \
    trainer.experiment_name='qwen_multiroute_distill_grpo_test' \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.val_before_train=False $@

train_exit=$?

# 清理 Ray 进程
[[ -n "${RAY_HEAD_PID:-}" ]] && kill "$RAY_HEAD_PID" 2>/dev/null || true

echo "=========================================="
echo "训练完成! 输出目录: $OUTPUT_DIR"
echo "日志文件: $VERL_FILE_LOGGER_PATH"
echo "退出码: $train_exit"
echo "=========================================="

exit ${train_exit}
