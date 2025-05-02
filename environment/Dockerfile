# --------- Stage 1: Build librealsense + Python deps ---------
  FROM python:3.10-slim-bookworm AS builder

  # 1) Predeclare so Docker’s linter won’t warn
  ENV LD_LIBRARY_PATH=/usr/local/lib
  ENV PYTHONPATH=/opt/app/lib/python3.10/site-packages
  
  # 2) Install all build tools & SDK prerequisites
  RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libssl-dev libusb-1.0-0-dev libudev-dev pkg-config \
        libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
        python3-dev python3-pip \
        libx11-dev libxkbcommon-dev \
      && rm -rf /var/lib/apt/lists/*
  
  # 3) Clone & build librealsense (no -j)
  RUN git clone https://github.com/IntelRealSense/librealsense.git /opt/librealsense \
   && mkdir /opt/librealsense/build \
   && cd /opt/librealsense/build \
   && cmake .. -DBUILD_PYTHON_BINDINGS=true \
   && make \
   && make install
  
  # 4) Install Python deps into a prefix
  WORKDIR /opt/app
  COPY requirements.txt .
  RUN pip3 install --prefix=/opt/app -r requirements.txt