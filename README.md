# MARIO
MARIO is an end-to-end Modular and Extensible Architecture for computing Visual Statistics in RoboCup SPL. 
MARIO ranked #1 at the RoboCup 2022 SPL Open Research Challenge.

[![MARIO GUI](https://img.youtube.com/vi/eutyWaQ4-oU/2.jpg)](https://www.youtube.com/watch?v=eutyWaQ4-oU)


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

 -Go to directory of repository 
 
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
     
## HOW TO USE MARIO 

Choose the video of the game to be analyzed and the game controller data from https://logs.naoth.de/2019-07-02_RC19-others/ and download.

	1) Upload the video via the choose video button;
	2) Upload game controller data corresponding to the game and team via the game controller button and select GPU/CPU via switch button;
	3) If the video has been calibrated ,that is, the fisheye distortion has been removed, switch the calibrated button otherwise via the choose 		     calibration button , select the calibration file within the calibration_data folder. In the name of the video file is the type of field in 	     which the robots play , just choose the corresponding calibration file;
	4) Click start calibration button;
	5) A progress bar will be shown in the terminal , when the process is finished , click the newly enabled button "go to tracking";
	6) After a background substraction process , two windows will appear for homography calculation. Red dots are shown in the real field, these dots 		will have to be located in the virtual field using the mouse and once selected just click 's' and go on for the next dots. Once the selection 	            of points is finished the tracking and analysis of the video will start.
	7) Finished tracking ,click the go analysis button;
	8) You can see the statistics by clicking on the various buttons that appear in the window. The heatmap and trackmap windows are closed by pressing 	          the 'q' key.

**NOTE:** 

- Fore more details and informations click [here](https://sites.google.com/unibas.it/wolves/robocup/robocup-2022/mario).
- For run MARIO in a **Docker Container** click [here](https://github.com/unibas-wolves/MARIO/tree/mario-docker).
	
