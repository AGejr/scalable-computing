FROM python:3.12.0

# Set the working directory
WORKDIR /app

# Copy the pre-downloaded font to the appropriate directory
COPY assets/Arial.ttf /root/.config/Ultralytics/Arial.ttf

# Copy your application files
COPY data /app/data
COPY models /app/models
COPY utils /app/utils
COPY train.py /app
COPY val.py /app
COPY requirements.txt /app
COPY yolov5n.yaml /app/yolov5n.yaml

# Install OpenGL libraries for OpenCV (libGL)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the default command
CMD ["python", "train.py", "--img", "416", "--batch", "16", "--epochs", "1", "--data", "data/data.yaml"]
