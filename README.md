## Realtime Facial Recognition

I started this project as a way to build an implementation of realtime facial recogniton for use in a company timeclock software & hardware solution.

I was planning, initially, on utilizing a Raspberry Pi Zero 2 W to run the software and a connected Arducam as the camera interface. However, it's becoming clear that this is configuration is very likely to change. Installing dlib on my pi02w took ~12 hours. Not practical at all.

# Supported Hardware and OS's

The camera hardware implementation has been written as an abstract factory, so that essentially you can use any type of camera hardware and any OS as long as it supports python. Currently, Darwin (macOS) and Linux (specifically Raspberry Pi Zero 2 W) have implemented classes in the CameraFactory.

### dlib and pi02w 

I have included steps in the installation instructions on how to build dlib with minimal requirements, as it was the only way I was able to get it installed.

# Linux Installation (Raspberry Pi Zero 2 W)

Run the following commands before installing the Python package requirements:

# Install system dependencies

    sudo apt update
    sudo apt install -y build-essential cmake pkg-config
    sudo apt install -y python3-dev python3-pip
    sudo apt install -y libatlas-base-dev gfortran
    sudo apt install -y libopenblas-dev

# Create and activate virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

# Install numpy first

    pip3 install numpy

# If installing on a Raspberry Pi Zero 2 (W), install picamera2

    pip install picamera2

# Increase swapfile space to prevent out-of-memory issues from dlib installation

    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon

# Clone dlib and build with minimal requirements

    git clone https://github.com/davisking/dlib.git
    cd dlib
    python3 setup.py install --no DLIB_USE_CUDA  

# Remove swapfile

    sudo swapoff -a
    sudo rm /swapfile

Edit fstab with `sudo nano /etc/fstab` and look for the line that contains `swapfile` and comment it out or remove it.

Double check the swapfile has been removed with `free -h`

# Install the required Python packages

    pip3 install -r requirements.txt  

# Create symlinks to the system python packages that we need to use in our virutal environment

    sudo ln -s /usr/lib/python3/dist-packages/libcamera* .venv/lib/python3.11/site-packages/
    sudo ln -s /usr/lib/python3/dist-packages/picamera2* .venv/lib/python3.11/site-packages/
    sudo ln -s /usr/lib/python3/dist-packages/pykms* .venv/lib/python3.11/site-packages/
    sudo ln -s /usr/lib/python3/dist-packages/kms* .venv/lib/python3.11/site-packages/
    sudo ln -s /usr/lib/python3/dist-packages/_pykms* .venv/lib/python3.11/site-packages/
