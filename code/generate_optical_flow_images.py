#!python
# -*- coding: utf-8 -*-

"""
Generate optical flow images for the sequences.

"""

from joblib import Parallel, delayed

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import time


def get_args():
    parser = argparse.ArgumentParser(description="Generate optical flow images.")

    parser.add_argument('--metadata-file',
                        required=True,
                        help="The path to the file containing the images metadata.")
    parser.add_argument('--images-dir',
                        required=True,
                        help="The path to the directory containing the images.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the output dir to which the optical flow images need to be written.")

    args = parser.parse_args()
    return args


def get_flow_rgb_images(sequence_frame_paths, flag=cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
    """
    Compute Dense Optical Flow using Gunner Farneback’s algorithm (BGR velocity image)
    computed from prev_frame and current_frame.

    :param sequence_frame_paths: List of the paths to the frames belonging to a sequence of images.
    :param flag: Chose from [0, cv2.OPTFLOW_USE_INITIAL_FLOW, cv2.OPTFLOW_FARNEBACK_GAUSSIAN].
        Default: cv2.OPTFLOW_FARNEBACK_GAUSSIAN.
    :return: (average_flow_rgb_image, list_of_all_flow_rgb_images)
        Note: The list_of_all_flow_rgb_images contains 'n-1' images if sequence_frame_names contains 'n' images.

    """
    # Initialize the first frame as reference frame.
    prev_frame = cv2.imread(sequence_frame_paths[0])
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Placeholders to store flow_rgb images and average flow_rgb image.
    flow_rgb_images = []
    avg_flow_rgb = np.zeros_like(prev_frame)
    num_flows = len(sequence_frame_paths) - 1  # There will be 1 less flow image as first image is a reference image.

    # Initialize HSV image to store and convert flow output to RGB image.
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # Produce flow_output_rgb images.
    for current_frame_name in sequence_frame_paths[1:]:
        current_frame_gray = cv2.cvtColor(cv2.imread(current_frame_name), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, flag)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        cur_flow_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)  # Actually BGR (not RGB) but essentially same information.

        flow_rgb_images.append(cur_flow_rgb)
        avg_flow_rgb = avg_flow_rgb + (cur_flow_rgb / num_flows)

        prev_frame_gray = current_frame_gray

    # Round off the average flow rgb image.
    avg_flow_rgb = np.array(np.round(avg_flow_rgb), dtype=np.uint8)

    return avg_flow_rgb, flow_rgb_images


def run_optical_flow_generation(data_row, images_dir, out_dir):
    seq_id = data_row["sequence"]
    image_paths = [os.path.join(images_dir, x) for x in (data_row.image1, data_row.image2, data_row.image3)]

    avg_flow_image, flow_images = get_flow_rgb_images(image_paths, flag=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # Save the flow images, and the average flow image.
    for i, cur_flow_image in enumerate(flow_images):
        cv2.imwrite(os.path.join(out_dir, "%s_opticalflowGF_%s.png" % (seq_id, i + 1)), cur_flow_image)

    cv2.imwrite(os.path.join(out_dir, "%s_opticalflowGF_%s.png" % (seq_id, "average")), avg_flow_image)


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset = pd.read_csv(args.metadata_file)

    start_time = time.time()
    Parallel(n_jobs=-1)(delayed(run_optical_flow_generation)(data_row,
                                                             args.images_dir,
                                                             args.out_dir)
                        for idx, data_row in dataset.iterrows())

    end_time = time.time()
    time_taken = end_time - start_time
    print("Completed generating optical flow images in %s seconds." % time_taken)
