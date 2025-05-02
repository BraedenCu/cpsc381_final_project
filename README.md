# CPSC381

## Environment setup

bash <(curl -L micro.mamba.pm/install.sh)

micromamba create -n realsense-stream -f environment.yml -c conda-forge

micromamba activate realsense-stream 

python -c "import pyrealsense2, flask, cv2; print('OK')"
