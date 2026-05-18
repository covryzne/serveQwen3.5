python -m vllm.entrypoints.openai.api_server \
  --model /workspace/shared/public/genericmodels/qwen_3.5_4b_noquantize \
  --dtype auto \
  --gpu-memory-utilization 0.5 \
  --max-model-len 16384 \
  --trust-remote-code \
  --port 8200