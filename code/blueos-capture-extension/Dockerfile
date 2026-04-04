FROM arm32v7/python:3.9-bullseye

RUN apt-get update && apt-get install -y \
    python3-opencv \
    python3-numpy \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first before the symlink verification
RUN pip install "numpy<2"

RUN ln -s $(find /usr/lib/python3 -name "cv2*.so" | head -1) /usr/local/lib/python3.9/site-packages/cv2.so \
    && python3 -c "import cv2; print('cv2 OK')"

RUN pip install flask
RUN mkdir -p /userdata
COPY main.py .
CMD ["python3", "main.py"]