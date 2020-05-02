Capstone Project Commands

# VM:
bhuvanvm: ssh msbhuvan@13.64.190.226
cpumachine-1: ssh vm@13.93.138.112
cpumachine-2: ssh vm@137.135.50.129

gpumachine-1: ssh vm@13.77.137.115
gpumachine-2: ssh vm@52.151.28.86

# Mounting the disk:
dmesg | grep SCSI

sudo fdisk /dev/<>
n
p
q

sudo mkdir /datadrive
sudo mount /dev/<>1 /datadrive

sudo -i blkid
sudo vi /etc/fstab
    UUID=<>   /datadrive   ext4   defaults,nofail   1   2

sudo reboot

ssh-keygen -t rsa -b 4096 -C "msbhuvanbhuvi@gmail.com"
cat .ssh/id_rsa.pub


# Connecting to Jupyter
with_ssh_tunnel_for_Jupyter: ssh -N -f -L localhost:8889:localhost:8889 vm@104.42.152.252
stop connection: sudo netstat -lpn |grep :8889 # and kill the shown PID.
starting jupyter: jupyter notebook --no-browser --port=8889 --allow-root &> jupyter.log
cat jupyter.log

# Docker.
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter [one time] # tensorflow/tensorflow:2.0.0-gpu-py3-jupyter
docker run -itd --gpus all -p 8889:8889 -e USER_HOME=$HOME/vm -v /datadrive:/datadrive tensorflow/tensorflow:latest-gpu-py3-jupyter bash
docker exec -it 4e267b28e70f /bin/bash  # Get the container_id: b619db070bb6, using docker ps.
cd ../datadrive/camera_trap_animal_classification/code


docker exec -it 3734e31aa5a6 /bin/bash  # gpumachine-2

# Things to install on docker
pip install pandas
pip install scikit-learn
pip install wget
apt-get install less

# Image Resizing:
from joblib import Parallel, delayed
from skimage import io
from skimage.transform import resize
import os
import time
from tqdm import tqdm

def resize_images_dir(src_dir, dest_dir, output_shape, njobs=1):
    def _resize_img(src_image_file):
        img = io.imread(os.path.join(src_dir, src_image_file))
        img_resized = resize(img, output_shape, anti_aliasing=True)
        img_resized = img_resized * 255  # The original resize function returns normalized image.
        img_resized = img_resized.astype('uint8')
        io.imsave(os.path.join(dest_dir, src_image_file), img_resized)

    start_time = time.time()
    os.makedirs(dest_dir)
    src_images = os.listdir(src_dir)
    num_images = len(src_images)
    print("Total number of images: %s" % num_images)
    Parallel(n_jobs=njobs)(delayed(_resize_img)(src_image_file) for src_image_file in tqdm(src_images))
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to resize %s images: %s seconds." % (num_images, time_taken))


resize_images_dir("/datadrive2/dataset/images", "/datadrive2/dataset/images-resized-224", (224,224) , njobs=16)

resize_images_dir("/datadrive2/dataset/images", "/datadrive2/dataset/images-resized", (512,512) , njobs=16)  # TODO Rename to: images-resized-512

#resize_images_dir("/datadrive2/dataset/images_test", "/datadrive2/dataset/images_test-resized", (224,224), njobs=16)

# Model training
The basic training code: https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

tf image summary issues: https://github.com/tensorflow/tensorflow/issues/28868

# Baseline modeling: mode_flat_all
# class imbalance:
{0: 10738, 1: 51426} -> {0:1, 1: 0.21}

### Test baseline training model.
python train_baseline_model.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline-trial --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline-trial.log 2>&1

python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline-trial-1 --model-arch vgg16_batchnorm --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 2 --learning-rate 0.001 --image-size 224

python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline-trial-2 --model-arch vgg16_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 2 --learning-rate 0.001 --image-size 224

python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline-trial-4 --model-arch resnet50_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 64 --epochs 1 --learning-rate 0.001 --image-size 224

python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline-trial-5 --model-arch inceptionresnetv2_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 64 --epochs 1 --learning-rate 0.001 --image-size 224

### Baseline model training.
python train_baseline_model.py --train-meta-file ../data/final_dataset_train.csv --val-meta-file ../data/final_dataset_val.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_1 --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline_1.log 2>&1 & [INCOMPLETE]

python train_baseline_model.py --train-meta-file ../data/final_dataset_train.csv --val-meta-file ../data/final_dataset_val.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_2 --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline_2.log 2>&1 & [INCOMPLETE]

python train_pipeline.py --train-meta-file ../data/final_dataset_train.csv --val-meta-file ../data/final_dataset_val.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_3 --model-arch vgg16_batchnorm --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline_3.log 2>&1 & [gpumachine-1, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train.csv --val-meta-file ../data/final_dataset_val.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_4 --model-arch vgg16_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline_4.log 2>&1 & [gpumachine-2, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_3_balanced --model-arch vgg16_batchnorm --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.001 --image-size 224 > ../logs/baseline_3_balanced.log 2>&1 & [gpumachine-2, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_4_balanced --model-arch vgg16_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_4_balanced.log 2>&1 & [gpumachine-2, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_5_balanced --model-arch resnet50_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 64 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_5_balanced.log 2>&1 & [gpumachine-3, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_6_balanced --model-arch inceptionresnetv2_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 64 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_6_balanced.log 2>&1 & [gpumachine-4, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_7_balanced --model-arch resnet101_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_7_balanced.log 2>&1 & [gpumachine-2, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_8_balanced --model-arch resnet152_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_8_balanced.log 2>&1 & [gpumachine-3, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_9_balanced --model-arch resnet152v2_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_9_balanced.log 2>&1 & [gpumachine-3, COMPLETED]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_8_updated --model-arch resnet152_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/baseline_8_updated.log 2>&1 & [gpumachine-3, COMPLETED]

# -------------------
# Darshan:
# Model: mask_background_mog2_single_balanced
python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask-background-trial-1 --model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --batch-size 16 --epochs 1 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask_MOG2_4channel_1 --model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/mask_MOG2_4channel_1.log 2>&1 & [gpumachine-4, IN PROCESS(132)]

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_4channel_1/val_auc --batch-size 32 --trained-model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --trained-checkpoint-dir ../trained_models/mask_MOG2_4channel_1/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-mask_MOG2_4channel_1-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/mask_MOG2_4channel_1/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/mask_MOG2_4channel_1/val_auc/evaluation/individual-roc

# Model: mask_background_mog2_multichannel
python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask-background-trial-2 --model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_multichannel --batch-size 16 --epochs 1 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask_MOG2_10channel_1 --model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_multichannel --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/mask_MOG2_10channel_1.log 2>&1 & [gpumachine-4, IN PROCESS(1283)]

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_10channel_1/val_auc --batch-size 32 --trained-model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_multichannel --trained-checkpoint-dir ../trained_models/mask_MOG2_10channel_1/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-mask_MOG2_10channel_1-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/mask_MOG2_4channel_1/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/mask_MOG2_4channel_1/val_auc/evaluation/individual-roc

# Train Optical Flow based models.
# Trial
python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_6channel_1-trial --model-arch resnet152_6channel_opticalflow --data-pipeline-mode mode_opticalflow_single --num-channels 6 --batch-size 8 --epochs 1 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005

python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_6channel_allpretrained_1-trial --model-arch resnet152_6channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflow_single --num-channels 6 --batch-size 8 --epochs 1 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005

# Model: opticalflow_6channel_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_6channel_1 --model-arch resnet152_6channel_opticalflow --data-pipeline-mode mode_opticalflow_single --num-channels 6 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_6channel_1.log 2>&1 & [gpumachine-2, COMPLETED]

# Model: opticalflow_6channel_allpretrained_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_6channel_allpretrained_1 --model-arch resnet152_6channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflow_single --num-channels 6 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_6channel_allpretrained_1.log 2>&1 & [gpumachine-3, COMPLETED]

# Model: opticalflow_15channel_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_15channel_1 --model-arch resnet152_15channel_opticalflow --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_15channel_1.log 2>&1 & [gpumachine-2, COMPLETED]

# Model: opticalflow_15channel_allpretrained_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_15channel_allpretrained_1 --model-arch resnet152_15channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_15channel_allpretrained_1.log 2>&1 & [gpumachine-3, COMPLETED]

# Train Hybrid Models.
# Model: hybrid_13channel_allpretrained_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/hybrid_13channel_allpretrained_1 --model-arch resnet152_13channel_allpretrained_hybrid --data-pipeline-mode mode_hybrid_13channel --num-channels 13 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/hybrid_13channel_allpretrained_1.log 2>&1 & [gpumachine-2, COMPLETED]

# Model: hybrid_16channel_allpretrained_1 (Very huge and runs out of the memory with shuffle buffer size of 10000, trying again, but if it fails, then reduce this to say 5000 and rerun).
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/hybrid_16channel_allpretrained_1 --model-arch resnet152_16channel_allpretrained_hybrid --data-pipeline-mode mode_hybrid_16channel --num-channels 16 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/hybrid_16channel_allpretrained_1.log 2>&1 & [gpumachine-3, COMPLETED]

# Train opticalflow or mask only models.
# Model: opticalflowonly_6channel_allpretrained_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflowonly_6channel_allpretrained_1 --model-arch resnet152_6channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflowonly_6channel --num-channels 6 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflowonly_6channel_allpretrained_1.log 2>&1 & [gpumachine-2, COMPLETED]

# Model: maskopticalflowonly_7channel_allpretrained_1
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/maskopticalflowonly_7channel_allpretrained_1 --model-arch resnet152_7channel_allpretrained_maskopticalflow --data-pipeline-mode mode_maskopticalflowonly_7channel --num-channels 7 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/maskopticalflowonly_7channel_allpretrained_1.log 2>&1 & [gpumachine-4, COMPLETED]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------


# Running inference pipeline. (specify --is-sequence-model flag for sequence models.)
# Model: baseline_3
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3/val_acc --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3/best_model_dir-acc.ckpt --extract-layers "block1_conv1-Relu,block3_conv3-Relu,block5_conv3-Relu" --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_3-val_acc.log 2>&1 &

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3/val_loss --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3/best_model_dir-loss.ckpt --extract-layers "block1_conv1-Relu,block3_conv3-Relu,block5_conv3-Relu" --image-size 224 > ../logs/inference-baseline_3-val_loss.log 2>&1 &

# Model: baseline_4
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4/val_acc --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4/best_model_dir-acc.ckpt --image-size 224 > ../logs/inference-baseline_4-val_acc.log 2>&1 &

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4/val_loss --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4/best_model_dir-loss.ckpt --image-size 224 > ../logs/inference-baseline_4-val_loss.log 2>&1 &

#---------
# Model: baseline_3_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3_balanced/val_acc --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3_balanced/best_model_dir-acc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_3_balanced-val_acc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_3_balanced/val_acc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_3_balanced/val_acc/evaluation/individual-roc

# Model: baseline_4_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4_balanced/val_acc --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4_balanced/best_model_dir-acc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_4_balanced-val_acc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_4_balanced/val_acc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_4_balanced/val_acc/evaluation/individual-roc

# Model: baseline_5_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_5_balanced/val_auc --batch-size 32 --trained-model-arch resnet50_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_5_balanced/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_5_balanced-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_5_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_5_balanced/val_auc/evaluation/individual-roc

# Model: baseline_6_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_6_balanced/val_auc --batch-size 32 --trained-model-arch inceptionresnetv2_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_6_balanced/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_6_balanced-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_6_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_6_balanced/val_auc/evaluation/individual-roc

# Model: baseline_7_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_7_balanced/val_auc --batch-size 32 --trained-model-arch resnet101_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_7_balanced/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_7_balanced-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_7_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_7_balanced/val_auc/evaluation/individual-roc

# Model: baseline_8_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_8_balanced/val_auc --batch-size 32 --trained-model-arch resnet152_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_8_balanced/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_8_balanced-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_8_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_8_balanced/val_auc/evaluation/individual-roc

# Model: baseline_9_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_9_balanced/val_auc --batch-size 32 --trained-model-arch resnet152v2_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_9_balanced/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_9_balanced-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_9_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_9_balanced/val_auc/evaluation/individual-roc


# Model: baseline_8_updated
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_8_updated/val_auc --batch-size 32 --trained-model-arch resnet152_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_8_updated/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_8_updated-val_auc.log 2>&1

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_8_updated/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_8_updated/val_auc/evaluation/individual-roc


#------
# Model: mask_MOG2_4channel_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_4channel_1/val_auc --batch-size 16 --trained-model-arch resnet152_mask_pretrained_imagenet --trained-checkpoint-dir ../trained_models/mask_MOG2_4channel_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_mask_mog2_single --num-channels 4 > ../logs/inference-mask_MOG2_4channel_1-val_auc.log 2>&1

# Model: mask_MOG2_10channel_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_10channel_1/val_auc --batch-size 16 --trained-model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --trained-checkpoint-dir ../trained_models/mask_MOG2_10channel_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_mask_mog2_multichannel --num-channels 10 > ../logs/inference-mask_MOG2_10channel_1-val_auc.log 2>&1


#------
# Model: opticalflow_6channel_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_6channel_1/val_auc --batch-size 16 --trained-model-arch resnet152_6channel_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_6channel_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_single --num-channels 6 > ../logs/inference-opticalflow_6channel_1-val_auc.log 2>&1

# Model: opticalflow_6channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_6channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_6channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_6channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_single --num-channels 6 > ../logs/inference-opticalflow_6channel_allpretrained_1-val_auc.log 2>&1

# Model: opticalflow_15channel_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_15channel_1/val_auc --batch-size 16 --trained-model-arch resnet152_15channel_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_15channel_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 > ../logs/inference-opticalflow_15channel_1-val_auc.log 2>&1

# Model: opticalflow_15channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_15channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_15channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_15channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 > ../logs/inference-opticalflow_15channel_allpretrained_1-val_auc.log 2>&1

#------
# Model: hybrid_13channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/hybrid_13channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_13channel_allpretrained_hybrid --trained-checkpoint-dir ../trained_models/hybrid_13channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_hybrid_13channel --num-channels 13 > ../logs/inference-hybrid_13channel_allpretrained_1-val_auc.log 2>&1

# Model: hybrid_16channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/hybrid_16channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_16channel_allpretrained_hybrid --trained-checkpoint-dir ../trained_models/hybrid_16channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_hybrid_16channel --num-channels 16 > ../logs/inference-hybrid_16channel_allpretrained_1-val_auc.log 2>&1


#-----
# Model: opticalflowonly_6channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflowonly_6channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_6channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflowonly_6channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflowonly_6channel --num-channels 6 > ../logs/inference-opticalflowonly_6channel_allpretrained_1-val_auc.log 2>&1

# Model: maskopticalflowonly_7channel_allpretrained_1
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/maskopticalflowonly_7channel_allpretrained_1/val_auc --batch-size 16 --trained-model-arch resnet152_7channel_allpretrained_maskopticalflow --trained-checkpoint-dir ../trained_models/maskopticalflowonly_7channel_allpretrained_1/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_maskopticalflowonly_7channel --num-channels 7 > ../logs/inference-maskopticalflowonly_7channel_allpretrained_1-val_auc.log 2>&1


### Optical Flow.
# Generate optical flow images [gpumachine-1]
python generate_optical_flow_images.py --metadata-file ../data/final_dataset_train.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../../wellington_data/images-resized-224-opticalflow/ &  [DONE]

python generate_optical_flow_images.py --metadata-file ../data/final_dataset_val.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../../wellington_data/images-resized-224-opticalflow/ &  [DONE]

python generate_optical_flow_images.py --metadata-file ../data/final_dataset_test.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../../wellington_data/images-resized-224-opticalflow/ &  [DONE]


#----------------------------------------

### Experiment with train-test splits based on the camera-sites.
# train set (up-sampled): loc_final_dataset_train_balanced-shuffled.csv (89777)	| unique: 53856	| unique sites: 111
# val set (down-sampled): loc_final_dataset_val_balanced-shuffled.csv (5969)	| unique: 16412	| unique sites: 36
# test set (down-sampled): loc_final_dataset_test_balanced-shuffled.csv (6453)	| unique: 18450	| unieuq sites: 35


### Train and Infererence the models:
# Model: baseline_8_balanced_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_8_balanced_loc --model-arch resnet152_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_8_balanced_loc.log 2>&1 & [gpumachine-2, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_8_balanced_loc/val_auc --batch-size 32 --trained-model-arch resnet152_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_8_balanced_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_flat_all > ../logs/inference-baseline_8_balanced_loc-val_auc.log 2>&1


# Model: resnet50_lstm_avg_pool_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/resnet50_lstm_avg_pool_loc --model-arch resnet50_pretrained_imagenet_lstm_avg_pool --data-pipeline-mode mode_sequence --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/resnet50_lstm_avg_pool_loc.log 2>&1 & [gpumachine-4, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/resnet50_lstm_avg_pool_loc/val_auc --batch-size 16 --trained-model-arch resnet50_pretrained_imagenet_lstm_avg_pool --trained-checkpoint-dir ../trained_models/resnet50_lstm_avg_pool_loc/ --filetype .h5 --image-size 224 --data-pipeline-mode mode_sequence > ../logs/inference-resnet50_lstm_avg_pool_loc-val_auc.log 2>&1


# Model: mask_MOG2_4channel_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask_MOG2_4channel_1_loc --model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --num-channels 4 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/mask_MOG2_4channel_1_loc.log 2>&1 & [gpumachine-2, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_4channel_1_loc/val_auc --batch-size 32 --trained-model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --trained-checkpoint-dir ../trained_models/mask_MOG2_4channel_1_loc/best_model_dir-auc.ckpt --image-size 224 --num-channels 4 > ../logs/inference-mask_MOG2_4channel_1_loc-val_auc.log 2>&1


# Model: mask_MOG2_10channel_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask_MOG2_10channel_1_loc --model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_multichannel --num-channels 10 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/mask_MOG2_10channel_1_loc.log 2>&1 & [gpumachine-3, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/mask_MOG2_10channel_1_loc/val_auc --batch-size 32 --trained-model-arch resnet152_mask_mog2_10channel_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_multichannel --trained-checkpoint-dir ../trained_models/mask_MOG2_10channel_1_loc/best_model_dir-auc.ckpt --image-size 224 --num-channels 10 > ../logs/inference-mask_MOG2_10channel_1_loc-val_auc.log 2>&1


# Model: opticalflow_6channel_allpretrained_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_6channel_allpretrained_1_loc --model-arch resnet152_6channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflow_single --num-channels 6 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_6channel_allpretrained_1_loc.log 2>&1 & [gpumachine-3, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_6channel_allpretrained_1_loc/val_auc --batch-size 16 --trained-model-arch resnet152_6channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_6channel_allpretrained_1_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_single --num-channels 6 > ../logs/inference-opticalflow_6channel_allpretrained_1_loc-val_auc.log 2>&1


# Model: opticalflow_15channel_allpretrained_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflow_15channel_allpretrained_1_loc --model-arch resnet152_15channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflow_15channel_allpretrained_1_loc.log 2>&1 & [gpumachine-4, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflow_15channel_allpretrained_1_loc/val_auc --batch-size 16 --trained-model-arch resnet152_15channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflow_15channel_allpretrained_1_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflow_multichannel --num-channels 15 > ../logs/inference-opticalflow_15channel_allpretrained_1_loc-val_auc.log 2>&1


# Model: hybrid_13channel_allpretrained_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/hybrid_13channel_allpretrained_1_loc --model-arch resnet152_13channel_allpretrained_hybrid --data-pipeline-mode mode_hybrid_13channel --num-channels 13 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/hybrid_13channel_allpretrained_1_loc.log 2>&1 & [gpumachine-4, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/hybrid_13channel_allpretrained_1_loc/val_auc --batch-size 16 --trained-model-arch resnet152_13channel_allpretrained_hybrid --trained-checkpoint-dir ../trained_models/hybrid_13channel_allpretrained_1_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_hybrid_13channel --num-channels 13 > ../logs/inference-hybrid_13channel_allpretrained_1_loc-val_auc.log 2>&1


# Model: opticalflowonly_6channel_allpretrained_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/opticalflowonly_6channel_allpretrained_1_loc --model-arch resnet152_6channel_allpretrained_opticalflow --data-pipeline-mode mode_opticalflowonly_6channel --num-channels 6 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/opticalflowonly_6channel_allpretrained_1_loc.log 2>&1 & [gpumachine-3, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/opticalflowonly_6channel_allpretrained_1_loc/val_auc --batch-size 16 --trained-model-arch resnet152_6channel_allpretrained_opticalflow --trained-checkpoint-dir ../trained_models/opticalflowonly_6channel_allpretrained_1_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_opticalflowonly_6channel --num-channels 6 > ../logs/inference-opticalflowonly_6channel_allpretrained_1_loc-val_auc.log 2>&1


# Model: maskopticalflowonly_7channel_allpretrained_1_loc
- training
python train_pipeline.py --train-meta-file ../data/loc_final_dataset_train_balanced-shuffled.csv --val-meta-file ../data/loc_final_dataset_val_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/maskopticalflowonly_7channel_allpretrained_1_loc --model-arch resnet152_7channel_allpretrained_maskopticalflow --data-pipeline-mode mode_maskopticalflowonly_7channel --num-channels 7 --batch-size 16 --epochs 10 --learning-rate 0.0001 --image-size 224 --patience 3 --min-delta-auc 0.005 > ../logs/maskopticalflowonly_7channel_allpretrained_1_loc.log 2>&1 & [gpumachine-4, COMPLETED]

- inference
python inference_pipeline.py --test-meta-file ../data/loc_final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/maskopticalflowonly_7channel_allpretrained_1_loc/val_auc --batch-size 16 --trained-model-arch resnet152_7channel_allpretrained_maskopticalflow --trained-checkpoint-dir ../trained_models/maskopticalflowonly_7channel_allpretrained_1_loc/best_model_dir-auc.ckpt --image-size 224 --data-pipeline-mode mode_maskopticalflowonly_7channel --num-channels 7 > ../logs/inference-maskopticalflowonly_7channel_allpretrained_1_loc-val_auc.log 2>&1


# ---------------------------------------
# Paper details:

### Sequence ids presented in sequence_example image.
left-top (animal, full daylight): 29125     |    info: 29125,"image1","110116151818047bs131.jpg","BIRD","047b","1/11/2016 15:18","s13"

right-top (animal, night time): 40156       |    info: 40156,"image1","060216224444038bs163.jpg","HEDGEHOG","038b","2/6/2016 22:44","s16"

left-bottom (animal, low daylight): 17068   |    info: 17068,"image1","160216060344039bs021.jpg","BIRD","039b","2/16/2016 6:03","s02"

right-bottom (empty, full daylight): 49305  |    info: 49305,"image1","010416125916018c5171.jpg","NOTHINGHERE","018c","4/1/2016 12:59","517"

# Data upload:
The data, trained models and inferene outputs uploaded to capstone_project container in storage account (resourcegroupcapstonedia).
Public Blob read-access SAS URL till 2021: https://resourcegroupcapstonedia.blob.core.windows.net/?sv=2019-10-10&ss=bfqt&srt=sco&sp=rl&se=2021-04-29T13:21:03Z&st=2020-04-29T05:21:03Z&spr=https&sig=rk8VhQ%2BTKhRmfSLAEgeUjmCFNBTksuSPkKYPBefbmrE%3D

# ArXiv details:
Title: Sequence Information Channel Concatenation for Improving Camera Trap Image Burst Classification

Authors: Bhuvan Malladihalli Shashidhara, Darshan Mehta, Yash Kale, Dan Morris, Megan Hazen

Abstract:
Camera Traps are extensively used to observe wildlife in their natural habitat without disturbing the ecosystem. This could help in the early detection of natural or human threats to animals, and help towards ecological conservation. Currently, a massive number of such camera traps have been deployed at various ecological conservation areas around the world, collecting data for decades, thereby requiring automation to detect images containing animals. Existing systems perform classification to detect if images contain animals by considering a single image. However, due to challenging scenes with animals camouflaged in their natural habitat, it sometimes becomes difficult to identify the presence of animals from merely a single image. We hypothesize that a short burst of images instead of a single image, assuming that the animal moves, makes it much easier for a human as well as a machine to detect the presence of animals. In this work, we explore a variety of approaches, and measure the impact of using short image sequences (burst of 3 images) on improving the camera trap image classification. We show that concatenating masks containing sequence information and the images from the 3-image-burst across channels, improves the ROC AUC by 20% on a test-set from unseen camera-sites, as compared to an equivalent model that learns from a single image.

Comments: 9 pages, 4 figures, 2 tables. Git repository can be found at: https://github.com/bhuvi3/camera_trap_animal_classification.

ACM-class: I.4.9; I.4.10; I.2.10

# ---------------------------------------

# XXX: Keep pushing this section towards the end. This summary might help later in future.
### Learnings (include mistakes and self-realizations):
- Adding channels helps in Improvements.
- Weak label models can be still trained, but consider deep models with residual layers, like ResNet152. Also, not much differnce from V1 and V2 (ResNet), maybe V1 is better due to pre-activation while batch-norm. For weak label models, Average Pool may help better than dense layers.
-  Optical flow Gunner Faurnebacks algorithm works great and similar to MOG2 background subtraction based motion detection for short sequences with objects moving in a relatively stationary environment.
- Pretraining almost always helps, and can be loaded by altering the intial layer with just a few lines of code. Additional channels can be initialized with random weights (or maybe with pretrained weights as well as these initial kernels learn very simple features which could be meaningful for any kind of mask or channel information).
- Be careful about 'training' argument in layers, and make sure that the inference is not run in training mode, as that leads to inconsistent predictions.
- Never ever, Never ever ever use "[[]] * x" syntax to initialize. Python will take you down unconsciously with its reference copies. Use list comprehension instead of ensure reference copy is avoided.
- Read TF and Keras documentation in detail, in all corners and crevices, they dont have good documentation. Also, consider reading their guides and examples before using.

### Improvements to consider for future projects.
# Code.
- Models could be maintained in configs. Or the model functions could take kwargs to put similar model generation together.


# Methods.
- Each method needs to get its optimal hyperparams like patience, but that takes a lot of time.
