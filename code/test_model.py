from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, ReLU, MaxPooling2D, Flatten, LSTM, Dropout, Conv2D, TimeDistributed


from data_pipeline import PipelineGenerator

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score
from tempfile import TemporaryFile

df = pd.read_csv("../data/final_dataset_test_balanced.csv")
new_model = tf.keras.models.load_model("../trained_models/baseline_4/best_model_dir-acc.ckpt/")


y_true = df.has_animal.values

prob = []
count = 1
accuracy_list = []
mean_abs_diff = []
labels = []
images = []

for index, row in df.iterrows():
    image1 = row['image1']
    image2 = row['image2']
    image3 = row['image3']
    label = row['has_animal']

    temp = []
    x = []
    c = []

    img1 = tf.io.read_file(tf.strings.join(["../../wellington_data/images-resized-224/", image1]))
    img1 = tf.image.decode_jpeg(img1, channels=3)
    img1 = tf.image.convert_image_dtype(img1, tf.float32).numpy()

    img2 = tf.io.read_file(tf.strings.join(["../../wellington_data/images-resized-224/", image2]))
    img2 = tf.image.decode_jpeg(img2, channels=3)
    img2 = tf.image.convert_image_dtype(img2, tf.float32).numpy()

    img3 = tf.io.read_file(tf.strings.join(["../../wellington_data/images-resized-224/", image3]))
    img3 = tf.image.decode_jpeg(img3, channels=3)
    img3 = tf.image.convert_image_dtype(img3, tf.float32).numpy()

    x.append(img1)
    #c.append(np.array(x))
    img1_prob = new_model.predict(np.array(x))[0][0]

    x = []

    x.append(img2)
    img2_prob = new_model.predict(np.array(x))[0][0]

    x = []
    x.append(img3)

    img3_prob = new_model.predict(np.array(x))[0][0]

    prob.append(img1_prob) 
    prob.append(img2_prob)
    prob.append(img3_prob)

    labels.append(label)
    labels.append(label)
    labels.append(label)

    images.append(image1)
    images.append(image2)
    images.append(image3)

    accuracy_list.append([img1_prob, label, image1])
    accuracy_list.append([img2_prob, label, image2])
    accuracy_list.append([img3_prob, label, image3])

    mean_abs_diff.append(abs(img1_prob-label))
    mean_abs_diff.append(abs(img2_prob-label))
    mean_abs_diff.append(abs(img3_prob-label))


    if count % 100 == 0:
        print(count)
        break
    count+=1


def calculate_avg_precision(y_true, y_scores):

        """
        Parameters:
        y_true: The true labels of the sequences
        y_scores: The scores associated with each sequence. The scores refer to the probability that the sequences have an animal
                  in atleast one of the images in it.
        """

        #Collect the different precision and recall values calculated at various probabilistic thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        precision_sum = 0
        count_precisions = 0
        
        for i in range(len(recalls)):
            if recalls[i] >=0.9:
                precision_sum+=precisions[i]
                count_precisions+=1

        #Take the mean of the precisions calculated for recall between 90-100
        average_precision = precision_sum/count_precisions
        return(average_precision)

#print(calculate_avg_precision(y_true, prob))
#print(accuracy_score(y_true, accuracy_list))
#print(np.mean(mean_abs_diff))

df_prob = pd.DataFrame(columns=['probs','true_label','image_id'])
df_prob['probs'] = prob
df_prob['true_label'] = labels
df_prob['image_id'] = images
df_prob.to_csv('test_dataset_stats.csv', index=False)

outfile = TemporaryFile()

np.save(outfile, accuracy_list)



