from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import random

class Test(object):

    """Takes in a set of images and returns the value of the chosen metric. The default metric is average precision at a recall\
      between 90-100"""

    """
    Parameters:
    
    predict_function: Takes in the name of predict function. The predict function depends on the type of model that is to be used.
                      This can take the predict function of three types of models: single-image, voting-based and sequence-based.
                      Default: There is no default function. The class must take one of the types listed above.
    metric: Takes the name of the metric to be calculated. Currently supports calculation of two metrics: average precision for recall
            between 90-100 and Area under ROC.
            Default is average precision

    """


    def __init__(self, predict_function, metric='average_precision'):
        
        self.predict_function = predict_function
        self.metric = metric
        

    def calculate_auc_roc(self, y_true, y_scores):

        """This function calculates the auc value and plots an ROC curve"""

        """
        Parameters:

        y_true: The true labels of the sequences

        y_scores: The scores associated with each sequence. The scores refer to the probability that the sequences have an animal
                  in atleast one of the images in it.
        """

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('AUC-ROC.png')
        return(roc_auc)

    def calculate_avg_precision(self, y_true, y_scores):

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

    def calculate_metric(self, images):

        """
        Parameters:

        images: The test set that needs to be labeled/predicted for the presence of an animal. It consists of a list of dictionaries
                with each entry containing the sequence of images and its associated labels.

        """

        y_true = []
        y_scores = []


        for image, label in images:

            image1 = image['image1'].numpy()
            image2 = image['image2'].numpy()
            image3 = image['image3'].numpy()

            y_true.append(label.numpy())

            y_score = self.predict_function(image1, image2, image3)
           
            y_scores.append(y_score)


        if self.metric == 'average_precision':
            average_precision = self.calculate_avg_precision(y_true, y_scores)
            return(average_precision)

        elif self.metric == 'auc_roc':
            auc_roc = self.calculate_auc_roc(y_true, y_scores)

            return(auc_roc)
