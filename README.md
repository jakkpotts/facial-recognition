# Linux Installation (Raspberry Pi Zero 2 W)

Run the following commands before installing the Python package requirements:

# Install system dependencies

    sudo apt update
    sudo apt install -y build-essential cmake pkg-config
    sudo apt install -y python3-dev python3-pip
    sudo apt install -y libatlas-base-dev gfortran
    sudo apt install -y libopenblas-dev

# Increase swap space to prevent out-of-memory issues

    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon

# Create and activate virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

# Install numpy first

    pip3 install numpy

# If running on a raspberry pi zero 2 (w), install picamera2

    pip install picamera2

# Clone dlib and build with minimal requirements

    git clone https://github.com/davisking/dlib.git
    cd dlib
    python3 setup.py install --no DLIB_USE_CUDA  

# Remove swapfile

    sudo swapoff -a
    sudo rm /swapfile
    sudo nano /etc/fstab
    free -h
    

# Install the required Python packages

    pip3 install -r requirements.txt  
