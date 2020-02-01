#%matplotlib qt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from roipoly1 import RoiPoly

import os


def label(file_name,train_folder):
    assert isinstance(file_name,str)
    assert isinstance(train_folder,str)

    file_path = 'C:/Users/shila/Desktop/Autograder/ECE276A- PR1/'+train_folder+'/'
    color_space = [[], [], []]
    for image in os.listdir(file_path):
        real_img = cv2.imread(file_path + image)
        img = cv2.cvtColor(real_img,cv2.COLOR_RGB2BGR)
        # Show the image
        fig = plt.figure()
        plt.imshow(img)
        plt.show(block=False)
        
        roi1 = RoiPoly(color='b', fig=fig)
    # Show the image with the first ROI
        fig = plt.figure()
        plt.imshow(img)
        roi1.display_roi()
        
        mask=roi1.get_mask(real_img[:,:,0])
        # Uncomment the lines below to see the mask image
        # plt.imshow(mask)
        # plt.show()

        pos = np.where(mask == True)
        img_YCrCb = cv2.cvtColor(real_img, cv2.COLOR_RGB2YCR_CB)

        Y_comp = img_YCrCb[:, :, 0]
        Cr_comp = img_YCrCb[:, :, 1]
        Cb_comp = img_YCrCb[:, :, 2]

        Y = Y_comp[pos]
        Cr = Cr_comp[pos]
        Cb = Cb_comp[pos]
    
        color_space[0].extend(Y.tolist())
        color_space[1].extend(Cr.tolist())
        color_space[2].extend(Cb.tolist()) 

    f = open('C:/Users/shila/Desktop/Autograder/ECE276A- PR1/Data_Label/'+file_name+'.pickle', 'wb')
    pickle.dump(color_space, f)
    f.close()

if __name__ == '__main__':
    print('Extracting Not_red pixel data')
    label('Not_STOP','Training_set_Not_Stop')
    print('Extracting red pixel data')
    label('STOP','Training_set_Stop')
    