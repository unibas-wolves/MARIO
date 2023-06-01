sudo apt-get update
sudo apt-get upgrade

sudo apt-get install build-essential cmake git pkg-config

# Without libtiff4-dev
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev

sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libatlas-base-dev gfortran

wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

sudo apt-get install python2.7-dev

pip install numpy

cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip

cd ~
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip

cd opencv-3.1.0/
mkdir build
cd build
# to build WITH CUDA remove option -D WITH_CUDA=OFF
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D WITH_CUDA=OFF \
	-D BUILD_EXAMPLES=ON ..
  
make -j4
sudo make install
sudo ldconfig

#python
#>>> import cv2

