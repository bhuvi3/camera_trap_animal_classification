# Exploring Sequences for Improving Camera Trap Image Classification
This was the capstone project for graduate students of the Master of Science in Data Science program at University of Washington, Seattle.
The main idea behind the problem statement is trying to exploit the information in sequences from camera trap images. The dataset contains sequences with 3 images in each sequence. Each sequence is associated with a label: '1' which means an animal is present in one or more images of the sequence, '0' which means there is no animal in all 3 images in the sequence. We form a baseline model which predicts whether an image has an animal or not i.e a single image classifier and then try to see if passing the sequence as an input to a different model helps increase our metric. 

### Getting Started
You should be able to clone this repository with the following command:
git clone https://github.com/bhuvi3/camera_trap_animal_classification.git

### Requirements
This project was run on a machine with a GPU. Basic requirements would include:
tensorflow version 2.1.0

All the experiments were run inside a docker that contained the latest tensorflow version along with a Jupyter notebook(not a mandatory requirement)

### Installing
To install and run the docker on a vm follow these steps:
1. Pull the docker image:
```
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter
```
2. Start the docker on port 8889:
```
docker run -itd --gpus all -p 8889:8889 -e USER_HOME=$HOME/<folder_to_run_docker> -v /folder_to_run_docker:/folder_to_run_docker tensorflow/tensorflow:latest-gpu-py3-jupyter bash
```
3. Run the docker ps command to get the image name

```
docker ps
```
4. Run the docker with the image name using the following command:
```
docker exec -it <docker_image_name> /bin/bash
```

If any of the above commands do not run, please run these commands as root user.

After successfull installation of the docker, run the following commands to install the python libraries inside the docker environment:
```
pip install pandas
pip install scikit-learn
pip install wget
apt-get install less
```
