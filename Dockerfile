FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

WORKDIR /workspace/EncoderOutcomePrediction

ENV PYTHONPATH=/workspace/EncoderOutcomePrediction
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir wandb==0.18.1 && \
    pip cache purge && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY . .

ENTRYPOINT ["python", "src/classification_main.py"]
