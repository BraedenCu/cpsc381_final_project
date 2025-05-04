# CPSC381

## Environment setup

bash <(curl -L micro.mamba.pm/install.sh)

micromamba create -n realsense-stream -f environment.yml -c conda-forge

micromamba activate realsense-stream 

python -c "import pyrealsense2, flask, cv2; print('OK')"


sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null


echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

## Downloading weights

gdown --id 1iYwkwxuIJ3xZhSePxcqW3TKX4LmLz_iI -O wave_sequence_model_final.h5

## Setting up environment on jetson

