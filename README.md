# MARIO with Docker

## PREREQUISITIES

- Ubuntu 20.04
- Docker

## INSTALL DOCKER

### Set up the repository

    $ sudo apt-get update

    $ sudo apt-get install ca-certificates curl gnupg lsb-release
    
### Add Docker’s official GPG key:

    $ sudo mkdir -p /etc/apt/keyrings
    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
 
### Use the following command to set up the repository:

    $ echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
### Install Docker Engine

    $ sudo apt-get update
    $ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    		
### Verify that Docker Engine is installed correctly by running the hello-world image

    $ sudo docker run hello-world
    		
### View docker images

    $ sudo docker images
    
# FOLDER SETUP

-Clone the following branch:

    $ sudo git clone --branch mario-docker https://github.com/unibas-wolves/MARIO.git
    
    **NOTE:** Rename this repository in "**MARIO-docker**"
    
 -Clone the following repository:  
 
    $ sudo git clone https://github.com/unibas-wolves/MARIO.git
    
-Copy **MARIO** folder to **MARIO-docker** folder:

    $ sudo cp -R ./MARIO ./MARIO-docker

-Insert the following files in the "**MARIO/detectionT**" folder:
  
  https://drive.google.com/drive/folders/1hDBn8gZZ1LzGC5JKYuN2AIpA_oewmNM2?usp=sharing

-Insert the following files in the "**MARIO/data**" folder: 

  https://drive.google.com/drive/folders/1jHWJbsgEpoFRs8ttHWARSFuaemOJ4BsJ?usp=sharing

-Insert the following files in the "**MARIO/video**" folder:
  
  https://drive.google.com/drive/folders/1Uea9DB4tz7uAb36V6ydfzTGwpQrn3YzJ?usp=sharing 
  
## BUILD DOCKER IMAGE WITH DOCKERFILE

-Open the terminal in the folder where the Dockerfile is located (in **MARIO-docker**) and type the following command:

    $ sudo docker build -t mario . 

-Run Docker Image:

    $ xhost +

    $ sudo docker run -it --volume=/tmp/.X11-unix:/tmp/.X11-unix  --device=/dev/dri:/dev/dri --env="DISPLAY=$DISPLAY" mario

    $ cd src/gui
    
    $ python3 mario.py
