# Exploring Sequences for Improving Camera Trap Image Classification
This was the capstone project for graduate students of the Master of Science in Data Science program at University of Washington, Seattle.
The main idea behind the problem statement is trying to exploit the information in sequences from camera trap images. The dataset contains sequences with 3 images in each sequence. Each sequence is associated with a label: '1' which means an animal is present in one or more images of the sequence, '0' which means there is no animal in all 3 images in the sequence. We form a baseline model which predicts whether an image has an animal or not i.e a single image classifier and then try to see if passing the sequence as an input to a different model helps increase our metric. 

## Detailed Report
The detailed report of this work has been published on arXiv in [this link (TODO)](TODO: add arXiv link). Please contact us for access to process data and results.

### Getting Started
You should be able to clone this repository with the following command:
```sh
git clone https://github.com/bhuvi3/camera_trap_animal_classification.git
```

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
3. Run the docker ps command to get the container name

```
docker ps
```
4. Run the docker with the container name using the following command:
```
docker exec -it <docker_container_name> /bin/bash
```

If any of the above commands do not run, please run these commands as root user.

After successful installation of the docker, run the following commands to install the python libraries inside the docker environment:
```
pip install pandas
pip install scikit-learn
pip install wget
apt-get install less
```
### TODO
How to install if not working on a vm

### Files
1. data_pipeline.py: This file returns a tensorflow dataset to be used during the training phase. This file has the definitions for the different modes in the training pipeline, depending on the training method to be used i.e single image mode or a sequence mode.

2. train_pipeline.py: This file initiates the training phase. Command line arguments are passed which mention what type of mode is to be used for training, training epochs, learning rate, etc all of which are mentioned in the file and shown in an example below

3. models.py: This file contains all the different models that have been tried and experimented with. 

4. inference_pipeline.py: This file runs the inference after training has been completed. It generates a folder which contains the ROC_AUC plot and stores the predictions of each sequence in a pickle file.

### Example
In the first run, we would advice to run a script that tests the data_pipeline.py code on a small trial dataset.

```
python train_pipeline.py --train-meta-file <final_dataset_train-trial.csv> --val-meta-file <final_dataset_val-trial.csv> --images-dir <images_directory>/ --out-dir <baseline-trial-1> --model-arch <mode_arch_name> --data-pipeline-mode <mode_name> --batch-size <batch_size> --epochs <epochs_count> --learning-rate <learning_rate> --image-size <image_size>
```
If that goes through without any errors, run the same script with the actual train dataset to train the model.
During the training phase in the sequence mode, if you get an error with the following error: [],[]...shape not correct, please re run the script again.

Run the inference script after training has completed

```
python inference_pipeline.py --test-meta-file <final_dataset_test_balanced-shuffled.csv> --images-dir <images_directory>/ --out-dir inference_outputs/baseline_4/val_acc --batch-size <batch_size> --trained-model-arch <model_arch_name> --trained-checkpoint-dir <trained_model_directory> --image-size <image_size> > <log_filename> 2>&1 &
```
