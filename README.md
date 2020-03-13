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
To install the docker on your local machine use the following command:
```
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter
```
