vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --max-model-len 108000 \
    --download-dir "/opt/dlami/nvme/hf"