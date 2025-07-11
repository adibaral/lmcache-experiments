LMCACHE_LOCAL_CPU=False \
LMCACHE_CONFIG_FILE="conf/lmcache_redis_config.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --max-model-len 108000 \
    --gpu-memory-utilization 0.9 \
    --download-dir "/opt/dlami/nvme/hf" \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1",
      "kv_role":"kv_both"
    }'