VLLM_LOGGING_CONFIG_PATH=logging_config.json \
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_REMOTE_URL="redis://localhost:6379" \
LMCACHE_REMOTE_SERDE="naive" \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --download-dir "/opt/dlami/nvme/hf-cache/hub/" \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1",
      "kv_role":"kv_both"
    }'

# meta-llama/Llama-3.1-8B-Instruct
# google/gemma-2-9b-it
# --log-config-file log_conf.json