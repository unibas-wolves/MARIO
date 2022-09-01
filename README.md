# MARIO
MARIO is an end-to-end Modular and Extensible Architecture for computing Visual Statistics in RoboCup SPL. 
MARIO ranked #1 at the RoboCup 2022 SPL Open Research Challenge.

## PREREQUISITIES

- Ubuntu 20.04
- Anaconda


## OpenCV INSTALLATION

### OpenCV from  ***source*** 

     sudo apt update

-Install the build tools and dependencies

     sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

-Clone the OpenCV’s and OpenCV contrib repositories:

     mkdir ~/opencv_build && cd ~/opencv_build

     git clone https://github.com/opencv/opencv.git

     git clone https://github.com/opencv/opencv_contrib.git

-Once the download is complete, create a temporary build directory, and navigate to it:

     cd ~/opencv_build/opencv

     mkdir -p build && cd build

-Set up the OpenCV build with CMake:

     cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

-Start the compilation process:

     make -j8

  NOTE: modify the -j flag according to your processor. 

-Install OpenCV with: 
  
     sudo make install

-To verify the installation, type the following commands and you should see the OpenCV version for Python bindings:

      python3 -c "import cv2; print(cv2.__version__)"

## CREATE CONDA ENVIROMENT

    conda create -n name_of_enviroment
    
    conda activate name_of_enviroment
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    git clone https://github.com/unibas-wolves/MARIO.git
 
     

## PYTHON LIBRARIES INSTALLATION 
 
     pip install -r requirements.txt
     
     pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip
     
## RUN PROJECT


-Insert the following files in the "**MARIO/detectionT**" folder:
  
  https://drive.google.com/drive/folders/1hDBn8gZZ1LzGC5JKYuN2AIpA_oewmNM2?usp=sharing

-Insert the following files in the "**MARIO/data**" folder: 

  https://drive.google.com/drive/folders/1jHWJbsgEpoFRs8ttHWARSFuaemOJ4BsJ?usp=sharing


-**RUN** project in "**MARIO/gui**" folder with following command:

     cd gui 
     
     python mario.py



**NOTE:** 

- Fore more details and informations click [here](https://sites.google.com/unibas.it/wolves/robocup/robocup-2022/mario).
- For run MARIO in a **Docker Container** click [here](https://github.com/unibas-wolves/MARIO/tree/mario-docker).
	
