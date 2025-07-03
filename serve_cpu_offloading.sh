LMCACHE_CONFIG_FILE=lmcache_config.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1",
      "kv_role":"kv_both"
    }'
