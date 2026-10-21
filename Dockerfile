# CPU image for the extraction API.
#
# Default (WITH_HF=0): the `replay` backend, which serves the fine-tuned model's recorded outputs. It's small (no
# torch) and starts instantly. CI builds it and smoke-tests the real HTTP path on every push.
#   docker build -t product-slm .
#   docker run -p 8000:8000 product-slm
#
# CPU fallback with the real model (WITH_HF=1): adds CPU-only torch + transformers + peft and mounts the adapter.
# It's slow (fp32 on CPU: seconds per title, unbatched) but runs anywhere. First start downloads the ~3 GB base model.
#   docker build --build-arg WITH_HF=1 -t product-slm:hf .
#   docker run -p 8000:8000 -v "$PWD/outputs/qwen2.5-1.5b-lora/adapter:/models/adapter:ro" \
#     -e SERVE_BACKEND=hf -e SERVE_MODEL=/models/adapter product-slm:hf
#
# GPU serving with vLLM is Dockerfile.gpu.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

ARG WITH_HF=0
# CPU wheels come from the PyTorch index. PyPI's Linux torch pulls in ~3 GB of CUDA libraries this image never uses.
RUN if [ "$WITH_HF" = "1" ]; then \
      uv pip install --python .venv/bin/python "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu && \
      uv pip install --python .venv/bin/python "transformers>=4.56" "peft>=0.13" "accelerate>=1.0"; \
    fi

COPY src ./src
COPY docs/LABELING.md ./docs/LABELING.md
COPY data/cassettes ./data/cassettes
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" SERVE_BACKEND=replay HF_HOME=/tmp/hf
RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "pslm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
