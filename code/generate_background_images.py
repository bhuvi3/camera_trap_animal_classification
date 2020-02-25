import os

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.image import imsave, imread


def save_mask(data_row, path_prefix, output_path):
    img1 = imread(os.path.join(path_prefix, data_row.image1))
    img2 = imread(os.path.join(path_prefix, data_row.image2))
    img3 = imread(os.path.join(path_prefix, data_row.image3))

    fgbg = cv2.createBackgroundSubtractorMOG2()

    frames = [img1, img2, img3]

    for frame in frames:
        fgmask = fgbg.apply(frame)
    
    imsave(os.path.join(output_path, 
                        str(data_row['sequence']) + '_mask_MOG2.png'), 
           fgmask, cmap='gray')
    

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(os.getcwd(), '..', 
                                       'data', 'final_dataset.csv'))
    path_prefix = os.path.join(os.getcwd(), '..', 'data', 
                               'images', 'images-resized')
    output_path = os.path.join(os.getcwd(), '..', 'data', 
                               'images', 'images-resized')
    
    Parallel(n_jobs=-1)(delayed(save_mask)(data_row, 
                                           path_prefix, 
                                           output_path) 
                        for idx, data_row in dataset.iterrows())
