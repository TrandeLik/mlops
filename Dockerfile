# --- Stage 1: Builder ---
FROM mambaorg/micromamba:1.5.8 as builder

WORKDIR /app

COPY environment.yaml .

RUN micromamba install -y -n base -f environment.yaml --channel-priority flexible && \
    micromamba clean --all --yes

COPY src/ ./src/
COPY predict.py ./predict.py
COPY models/final_model/ ./models/final_model


# --- Stage 2: Final Image ---
FROM mambaorg/micromamba:1.5.8

WORKDIR /app

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app/models/ ./models/
COPY --from=builder /app/src/ ./src/
COPY --from=builder /app/predict.py ./predict.py

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-m", "predict"]
