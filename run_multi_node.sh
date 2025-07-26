# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG

function default_value() {
        if [ -z "$1" ]; then
                echo "$2"
        else
                echo "$1"
        fi
}

# Hyperparameter =========
BS=$(default_value $BS 128)
GPUS=$(default_value $GPUS $(nvidia-smi --list-gpus | wc -l))
NUM_GROUPS=$(default_value $WORLD_SIZE 1)
GROUP_RANK=$(default_value $RANK 0)
MASTER_ADDR=$(default_value $MASTER_ADDR "localhost")
MASTER_PORT=$(default_value $MASTER_PORT 12345)
TP=$(default_value $TP 4)


# Notes ==========
# Setting `--save_steps 16` to save checkpoints every 16 policy iteration steps.
# Set `--eval_opponent_names google/gemini-2.0-flash-lite-001` if you have OpenRouter access.

python train_spiral.py \
    --env_id KuhnPoker-v1 \
    --use_llm_obs_wrapper \
    --eval_env_ids TicTacToe-v0 KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers False True \
    --eval_opponent_names google/gemini-2.0-flash-lite-001 \
    --eval_split all \
    --gamma 1 \
    --gpus $GPUS \
    --num_gpus_per_actor $TP \
    --num_groups $NUM_GROUPS \
    --group_rank $GROUP_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --gradient-checkpointing \
    --num_samples 1 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --pretrain Qwen/Qwen3-14B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --zero_stage 3 \
    --no-use_fused_lm_head \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --rollout_batch_size $BS \
    --rollout_batch_size_per_device $(( $BS / $GPUS / $NUM_GROUPS )) \
    --pi_buffer_maxlen_per_device $(( $BS / $GPUS / $NUM_GROUPS )) \
    --train_batch_size $BS \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --max_context_length 32768 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 16 \
    --save_steps -1 \
    --eval_games 16 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 51200 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name spiral-qwen3-14b-base-kp-4k-self-play \
    --wb_project spiral
