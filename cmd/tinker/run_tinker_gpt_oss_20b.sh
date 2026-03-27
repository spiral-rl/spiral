#!/bin/bash
# SPIRAL Tinker addon — GPT-OSS-20B training script
#
# Usage:
#   bash cmd/tinker/run_tinker_gpt_oss_20b.sh
#
# Required environment variables:
#   TINKER_API_KEY      — Tinker service API key
#   WANDB_API_KEY       — Weights & Biases API key
#   OPENROUTER_API_KEY  — OpenRouter API key (for LLM eval opponents)

set -euo pipefail

# ── Validate environment ─────────────────────────────────────────────────────
for var in TINKER_API_KEY WANDB_API_KEY OPENROUTER_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: $var is not set"
        echo "  export $var=your_key"
        exit 1
    fi
done

export WEAVE_PRINT_CALL_LINK=false

# ── Configurable defaults (override via env vars) ────────────────────────────
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
RENDERER="${RENDERER:-gpt_oss}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN="${NUM_TRAIN:-51200}"
LR="${LR:-4e-5}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ENV_IDS="${ENV_IDS:-TicTacToe-v0,KuhnPoker-v1,SimpleNegotiation-v2}"
OBS_WRAPPERS="${OBS_WRAPPERS:-False,True,True}"
EVAL_OPPONENTS="${EVAL_OPPONENTS:-google/gemini-2.0-flash-001}"
WANDB_PROJECT="${WANDB_PROJECT:-spiral}"
WANDB_NAME="${WANDB_NAME:-gpt-oss-20b-tinker-addon}"

# ── Launch training ──────────────────────────────────────────────────────────
python train_spiral_tinker.py \
    model_name="$MODEL_NAME" \
    renderer_name="$RENDERER" \
    lora_rank="$LORA_RANK" \
    env_ids="$ENV_IDS" \
    use_llm_obs_wrappers="$OBS_WRAPPERS" \
    batch_size="$BATCH_SIZE" \
    num_train_datapoints="$NUM_TRAIN" \
    num_test_datapoints=128 \
    learning_rate="$LR" \
    max_tokens="$MAX_TOKENS" \
    num_substeps=1 \
    use_role_baseline=True \
    role_baseline_ema_gamma=0.95 \
    eval_env_ids="$ENV_IDS" \
    eval_use_llm_obs_wrappers="$OBS_WRAPPERS" \
    eval_opponent_names="$EVAL_OPPONENTS" \
    eval_every=16 \
    save_every=20 \
    loss_fn=importance_sampling \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    use_streaming=false
