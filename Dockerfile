# downloader stage is used to obtain the model code from provate repo + model weights from Zenodo
FROM ubuntu AS downloader
WORKDIR /work
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates wget unzip

RUN mkdir /app
# downloading pretrained weights from Zenodo
RUN wget https://zenodo.org/record/6663662/files/head_swin_bnneck.zip -O /app/head_swin_bnneck.zip
RUN unzip /app/head_swin_bnneck.zip -d /app/head_swin_bnneck
RUN rm /app/head_swin_bnneck.zip
# last.ckpt is the same as model.ckpt, so deleting it to save image space
RUN rm /app/head_swin_bnneck/last.ckpt

FROM python:3.9-slim AS FINAL

# installing openCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /requirements.txt

# --extra-index-url https://download.pytorch.org/whl/cpu avoids CUDA installation
RUN python -m pip install --upgrade pip && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /requirements.txt
COPY --from=downloader /app .

ENV KAFKA_URL=kafka:9092
ENV INPUT_QUEUE=kashtanka_calvin_zhirui_yolov5_output
ENV OUTPUT_QUEUE=kashtanka_calvin_zhirui_embeddings_output
CMD python3 serve.py
COPY code .

FROM FINAL as TESTS
COPY example /app/example
RUN python -m unittest discover -v

