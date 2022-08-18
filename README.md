# MARIO
MARIO is an end-to-end Modular and extensible ARchitecture for computing vIsual statistics in rObocup spl. MARIO ranked #1 at the robocup 2022 SPL open research challenge.

## PREREQUISITIES

- Ubuntu 20.04

## UBUNTU DEPENDENCY INSTALLATION

    $ sudo apt update

    $ sudo apt install zlib1g-dev libjpeg-dev libpng-dev

    $ sudo apt install python3-pip

    $ sudo apt-get install git

## OpenCV INSTALLATION

### OpenCV from Python ***PyPi*** 

    $ pip install opencv-python==4.6.0.66

### OpenCV from  ***source*** 

    $ sudo apt update

-Install the build tools and dependencies

    $ sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

-Clone the OpenCV’s and OpenCV contrib repositories:

    $ mkdir ~/opencv_build && cd ~/opencv_build

    $ git clone https://github.com/opencv/opencv.git

    $ git clone https://github.com/opencv/opencv_contrib.git

-Once the download is complete, create a temporary build directory, and navigate to it:

    $ cd ~/opencv_build/opencv

    $ mkdir -p build && cd build

-Set up the OpenCV build with CMake:

    $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

-Start the compilation process:

    $ make -j8

  NOTE: modify the -j flag according to your processor. 

-Install OpenCV with: 
  
    $ sudo make install

-To verify the installation, type the following commands and you should see the OpenCV version for Python bindings:

    $  python3 -c "import cv2; print(cv2.__version__)"


## Tkinter INSTALLATION

    $ sudo apt-get install python3-tk

## PYTHON LIBRARIES INSTALLATION 
 
    $ pip install requests==2.28.0 
    
    $ pip install torch==1.11.0 torchaudio==0.11.0 torchvision==0.12.0
    
    $ pip install pyyaml==6.0
    
    $ pip install tqdm==4.64.0

    $ pip install matplotlib==3.5.2

    $ pip install seaborn==0.11.2

    $ pip install gdown==4.4.0

    $ pip install cython==0.29.30

    $ pip install tensorboard==2.9
    
    $ pip install easydict==1.9

    $ pip install scikit-learn==1.1.1

    $ pip install protobuf==3.20.0
    
    $ pip install Pillow==9.2.0

    $ pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip	

    $ pip installpython-math
    
    $ pip install statistics
    
    $ pip install ezprogress

## CLONE AND RUN PROJECT

    $ sudo git clone https://github.com/unibas-wolves/MARIO.git

-Insert the following files in the "**MARIO/detectionT**" folder:
  
  https://drive.google.com/drive/folders/1hDBn8gZZ1LzGC5JKYuN2AIpA_oewmNM2?usp=sharing

-Insert the following files in the "**MARIO/data**" folder: 

  https://drive.google.com/drive/folders/1jHWJbsgEpoFRs8ttHWARSFuaemOJ4BsJ?usp=sharing

-Insert the following files in the "**MARIO/video**" folder:
  
  https://drive.google.com/drive/folders/1Uea9DB4tz7uAb36V6ydfzTGwpQrn3YzJ?usp=sharing

-**RUN** project in "**MARIO**" folder with following command:

    $ python3 ./gui/mario.py

### Possible problems at this step:

1. **FileNotFoundError: No such file or directory: 'python'**

    Change the value "***python***" in "***python3***" in "**Preparation.py**" in line 70
    
2. **FileExistsError: File exists: './MARIO/imbs-mt/images'**

    Delete "**images**" folder in "**MARIO/imbs-mt**" folder
     

**NOTE:** Fore more details and informations click [here](https://sites.google.com/unibas.it/wolves/robocup/robocup-2022/mario).	
