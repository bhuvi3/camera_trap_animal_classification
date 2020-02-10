Capstone Project Commands

# VM:
bhuvanvm: ssh msbhuvan@13.64.190.226
cpumachine-1: ssh vm@13.93.138.112
cpumachine-2: ssh vm@137.135.50.129
gpumachine-1: ssh vm@52.175.247.163

# Mounting the disk:
dmesg | grep SCSI # get <>
sudo mkdir /datadrive2
sudo mount /dev/<> /datadrive2

sudo -i blkid # get <>
sudo vi /etc/fstab
    UUID=7b3f9aae-77f2-471e-aa34-b93fd5c66afd   /datadrive2   ext4   defaults,nofail   1   2

save and restart (sudo reboot)

# Connecting to Jupyter
with_ssh_tunnel_for_Jupyter: ssh -N -f -L localhost:8889:localhost:8889 vm@104.42.152.252
stop connection: sudo netstat -lpn |grep :8889 # and kill the shown PID.
starting jupyter: jupyter notebook --no-browser --port=8889 --allow-root &> jupyter.log
cat jupyter.log

# Docker.
docker run -itd --gpus all -p 8889:8889 -e USER_HOME=$HOME/vm -v /datadrive:/datadrive tensorflow/tensorflow:latest-gpu-py3-jupyter bash [Currently running, no need to run this again]
docker exec -it d6b147aba1f7 /bin/bash  # Get the container_id
cd ../datadrive/camera_trap_animal_classification


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

### Baseline model training.
python train_basline_model.py --train-meta-file ../data/final_dataset_train.csv --val-meta-file ../data/final_dataset_val.csv --images-dir ../data/images-resized-224/ --out-dir ../trained_models/baseline_1 --batch-size 64 --epochs 100 --learning-rate 0.001 --image-size 224
