# CPSC381

## Structure of Repository

### Inference + Training Files

webcam_hosted.py --> specify path to weights, hosts display of the model in action in your browswer
odometry_hosted.py --> for usage on the drone, displays model alongside odometry statistics
training/drone_training.ipynb --> full final training pipeline for creating the model

### Misc Testing Files

All other .ipynb / .py files in the repository. We do not garuntee correctness of any of these files, as they were only temporarily utilized to isolate performance of certain components of our architecture.


## Environment setup: MacOS

bash <(curl -L micro.mamba.pm/install.sh)

micromamba create -n realsense-stream -f environment.yml -c conda-forge

micromamba activate realsense-stream 

python -c "import pyrealsense2, flask, cv2; print('OK')"

pip install -r requirements.txt

python webcam_hosted.py

## Environment setup: Linux (Experimental)

sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

## Downloading weights

1. Get the ID from the share link created when you attempt to share uploaded links from google drive: https://drive.google.com/file/d/ID/view

2. using the ID (note: it is between /view and /file/d) plug it into the command in 4. Repace "ID" with your ID from the gdrive link as specified in step 1.

3. run pip install gdown

4. gdown --id ID -O wave_sequence_model_final.keras
