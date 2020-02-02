File Description:

stop_sign_detector.py : Main program which creates the segmetation mask and detects the stop signs in an image by drawing a green color bounding box around it.

roipoly1.py : This file was imported into hand_label.py file to create a Region of Interest on the image and extract red and not red pixels to build the data set for training.

train.py : This file was used to find train and find the classifer parameters for the red and not red class. The evaluated parameters were stored in a numpy arrary as:
                mu_red.npy- Mean of Red pixel class
                mu_not_red.py- Mean of Not-Red Class
                cov_not_red.npy- Covariance of Not-Red Class
                cov_red.npy- Covariance of Red Class
                P_red.npy- Probability of Red Class
                P_not_red.npy- Probability of Not-Red class

hand_label.py : This file was used to select the red and not red pixels and label them as red and not red. The roiploy1.py file was imorted into this file to select the pixels and build the test dataset.
