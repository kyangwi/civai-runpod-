FROM vllm/vllm-openai:v0.19.0

WORKDIR /app
RUN uv pip install --system --no-cache runpod==1.8.1 sentencepiece==0.2.1 \
    && uv pip check --system

ENV PYTHONUNBUFFERED=1 \
    QWEN_MODEL=Qwen/Qwen3.5-9B \
    MAX_MODEL_LEN=8192 \
    HF_HOME=/runpod-volume/huggingface

COPY rp_handler.py /app/rp_handler.py
ENTRYPOINT ["python3", "-u", "/app/rp_handler.py"]
