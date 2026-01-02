FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn torch --index-url https://download.pytorch.org/whl/cpu

COPY webapp/ ./webapp/
COPY ryan_gpt_basics/ ./ryan_gpt_basics/
COPY ryan_gpt_systems/ ./ryan_gpt_systems/
COPY ryan_gpt_data/ ./ryan_gpt_data/

RUN mkdir -p models/pretrain models/finetune && \
    curl -L -o models/pretrain/ckpt_final.pt "https://huggingface.co/ryanzhou147/ryan-gpt/resolve/main/pretrain_wikipedia/ckpt_final.pt" && \
    curl -L -o models/finetune/ckpt_final.pt "https://huggingface.co/ryanzhou147/ryan-gpt/resolve/main/finetune_dailydialog/ckpt_final.pt"

EXPOSE 8080
CMD ["python", "webapp/app.py", "--host", "0.0.0.0", "--port", "8080"]
