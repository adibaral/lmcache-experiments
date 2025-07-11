#!/bin/bash

# Set variables
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
HOST="http://localhost"
PORT=8000
SYSTEM_PROMPT="You are a helpful assistant."
MODE="query"
PROMPT_FILE="data/queries.txt"
# TRUNCATE="--truncate-context"
CONTEXT_FILES=(
    "data/conversation_history/conversation_history_0.json"
    "data/conversation_history/conversation_history_1.json"
    "data/conversation_history/conversation_history_2.json"
    "data/conversation_history/conversation_history_3.json"
    "data/conversation_history/conversation_history_4.json"
    "data/conversation_history/conversation_history_5.json"
    "data/conversation_history/conversation_history_6.json"
    "data/conversation_history/conversation_history_7.json"
    "data/conversation_history/conversation_history_8.json"
)
MAX_CTX_TOKENS=10800
SAFETY_MARGIN=2048
OUTPUT_FILE="benchmark_redis_query.json"
SEED=42

# Build the context file arguments
CONTEXT_ARGS=""
for FILE in "${CONTEXT_FILES[@]}"; do
    CONTEXT_ARGS+=" $FILE"
done

# Run the benchmark
uv run python src/benchmark.py \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --system-prompt "$SYSTEM_PROMPT" \
    --mode "$MODE" \
    --prompt-file "$PROMPT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --seed "$SEED" \
    --context-files $CONTEXT_ARGS \
    --max-ctx-tokens "$MAX_CTX_TOKENS" \
    --safety-margin "$SAFETY_MARGIN" \
    # --truncate-context \
