FROM python:3.12.0

WORKDIR /app

COPY models /app/models
COPY utils /app/utils
COPY yolo.py /app
COPY requirements.txt /app
COPY yolov5n.yaml /app/yolov5n.yaml

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "yolo.py"]