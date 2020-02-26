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

#--------
python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_7_balanced --model-arch resnet101_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_7_balanced.log 2>&1 & [gpumachine-2, IN PROCESS(35)]

python train_pipeline.py --train-meta-file ../data/final_dataset_train_balanced.csv --val-meta-file ../data/final_dataset_val_balanced.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/baseline_8_balanced --model-arch resnet152_pretrained_imagenet --data-pipeline-mode mode_flat_all --batch-size 32 --epochs 10 --learning-rate 0.0001 --image-size 224 > ../logs/baseline_8_balanced.log 2>&1 & [gpumachine-3, IN PROCESS(42)]

#--------


# Running inference pipeline. (specify --is-sequence-model flag for sequence models.)
# Model: baseline_3
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3/val_acc --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3/best_model_dir-acc.ckpt --extract-layers "block1_conv1-Relu,block3_conv3-Relu,block5_conv3-Relu" --image-size 224 > ../logs/inference-baseline_3-val_acc.log 2>&1 &

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3/val_loss --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3/best_model_dir-loss.ckpt --extract-layers "block1_conv1-Relu,block3_conv3-Relu,block5_conv3-Relu" --image-size 224 > ../logs/inference-baseline_3-val_loss.log 2>&1 &

# Model: baseline_4
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4/val_acc --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4/best_model_dir-acc.ckpt --image-size 224 > ../logs/inference-baseline_4-val_acc.log 2>&1 &

python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4/val_loss --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4/best_model_dir-loss.ckpt --image-size 224 > ../logs/inference-baseline_4-val_loss.log 2>&1 &

#---------
# Model: baseline_3_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_3_balanced/val_acc --batch-size 32 --trained-model-arch vgg16_batchnorm --trained-checkpoint-dir ../trained_models/baseline_3_balanced/best_model_dir-acc.ckpt --image-size 224 > ../logs/inference-baseline_3_balanced-val_acc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_3_balanced/val_acc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_3_balanced/val_acc/evaluation/individual-roc.png

# Model: baseline_4_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_4_balanced/val_acc --batch-size 32 --trained-model-arch vgg16_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_4_balanced/best_model_dir-acc.ckpt --image-size 224 > ../logs/inference-baseline_4_balanced-val_acc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_4_balanced/val_acc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_4_balanced/val_acc/evaluation/individual-roc.png

# Model: baseline_5_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_5_balanced/val_auc --batch-size 32 --trained-model-arch resnet50_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_5_balanced/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-baseline_5_balanced-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_5_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_5_balanced/val_auc/evaluation/individual-roc.png

# Model: baseline_6_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_6_balanced/val_auc --batch-size 32 --trained-model-arch inceptionresnetv2_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_6_balanced/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-baseline_6_balanced-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_6_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_6_balanced/val_auc/evaluation/individual-roc.png

# Model: baseline_7_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_7_balanced/val_auc --batch-size 32 --trained-model-arch resnet101_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_7_balanced/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-baseline_7_balanced-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_7_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_7_balanced/val_auc/evaluation/individual-roc

# Model: baseline_8_balanced
python inference_pipeline.py --test-meta-file ../data/final_dataset_test_balanced-shuffled.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../inference_outputs/baseline_8_balanced/val_auc --batch-size 32 --trained-model-arch resnet152_pretrained_imagenet --trained-checkpoint-dir ../trained_models/baseline_8_balanced/best_model_dir-auc.ckpt --image-size 224 > ../logs/inference-baseline_8_balanced-val_auc.log 2>&1 &

python compute_roc.py --preds-labels-file ../inference_outputs/baseline_8_balanced/val_auc/pred_labels-individual.pickle --out-file ../inference_outputs/baseline_8_balanced/val_auc/evaluation/individual-roc

# -------------------
# Model: mask_background_mog2_single_balanced
python train_pipeline.py --train-meta-file ../data/final_dataset_train-trial.csv --val-meta-file ../data/final_dataset_val-trial.csv --images-dir ../../wellington_data/images-resized-224/ --out-dir ../trained_models/mask-background-trial-1 --model-arch resnet152_mask_pretrained_imagenet --data-pipeline-mode mode_mask_mog2_single --batch-size 64 --epochs 1 --learning-rate 0.001 --image-size 224