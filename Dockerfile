FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG REPO_WS=/MARIO
RUN mkdir -p $REPO_WS/src
WORKDIR /home/user/$REPO_WS

RUN apt-get update

RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgtk2.0-dev zlib1g-dev libjpeg-dev libpng-dev

RUN apt install -y git

RUN apt-get install -y python3.8

RUN apt install python3-pip -y

RUN pip install opencv-python==4.6.0.66

RUN apt-get install -y python3-tk

RUN pip install requests==2.28.0

RUN pip install torch==1.11.0 torchaudio==0.11.0 torchvision==0.12.0

RUN pip install pyyaml==6.0

RUN pip install tqdm==4.64.0

RUN pip install matplotlib==3.5.2

RUN pip install seaborn==0.11.2

RUN pip install gdown==4.4.0

RUN pip install cython==0.29.30

RUN pip install tensorboard==2.9.

RUN pip install easydict==1.9

RUN pip install scikit-learn==1.1.1

RUN pip install protobuf==3.20.0

RUN pip install Pillow==9.2.0

RUN pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip

RUN pip install python-math

RUN pip install statistics

RUN pip install ezprogress

COPY ./MARIO /home/user/MARIO/src
