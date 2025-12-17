# Guide on how to install all needed packages for yolo11n_simple.py script

sudo apt update && sudo apt upgrade -y
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install ultralytics

### On my side I had to reinstall simplejpeg
pip uninstall -y simplejpeg
pip install --no-cache-dir --force-reinstall simplejpeg

### run the script
python yolo11n_simple.py



