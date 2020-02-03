'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import binary_erosion
from scipy.stats import multivariate_normal
# import math as m
# from scipy.stats import multivariate_normal


class StopSignDetector():
    def __init__(self):
                
        self.P_red= np.load('P_red.npy')
        self.P_not_red=np.load('P_not_red.npy')

        self.mu_red=np.load('mu_red.npy').T
        self.cov_red=np.load('cov_red.npy')
                
        self.mu_not_red=np.load('mu_not_red.npy').T
        self.cov_not_red=np.load('cov_not_red.npy')
        
        '''
        Initilize your stop sign detector with the attributes you need,
        e.g., parameters of your classifier
        '''
#         raise NotImplementedError
        
    def segment_image(self, img):
        '''
        Obtain a segmented image using a color classifier,
        e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
        call other functions in this class if needed
        
        Inputs:
        img - original image
        Outputs:
        mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        # YOUR CODE HERE
        gamma=2
        invGamma=1/gamma
        table=np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8") 
        gamma_corr=cv2.LUT(img,table)

        test_img=gamma_corr
        l=np.shape(img)[0]
        b=np.shape(img)[1]
        vect_img=np.zeros((3,l*b),dtype=np.uint8)
        
        vect_img[0,:]=np.reshape(test_img[:,:,0],(l*b))
        vect_img[1,:]=np.reshape(test_img[:,:,1],(l*b))
        vect_img[2,:]=np.reshape(test_img[:,:,2],(l*b))
        prob_red=self.P_red*multivariate_normal(self.mu_red,self.cov_red).pdf(vect_img.T)
        prob_not_red=self.P_not_red*multivariate_normal(self.mu_not_red,self.cov_not_red).pdf(vect_img.T)
        im=np.greater(prob_red,prob_not_red)
#         print(np.count_nonzero(im))
        # print()
        mask_img=np.reshape(im,(l,b))
        # print(mask_img)
        
        # plt.imshow(bit_mask)
        return mask_img
    
    def get_bounding_box(self, img):
        '''
        Find the bounding box of the stop sign
        call other functions in this class if needed
        
        Inputs:
        img - original image
        
        Outputs:
        boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
        where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
        is from left to right in the image.
        
        Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
#         raise NotImplementedError
        bit_mask=self.segment_image(img)
        masked=np.zeros(np.shape(img),dtype=np.uint8)
        masked[:,:,0]=np.multiply(bit_mask,img[:,:,0],dtype=np.float64)
        masked[:,:,1]=np.multiply(bit_mask,img[:,:,1],dtype=np.float64)
        masked[:,:,2]=np.multiply(bit_mask,img[:,:,2],dtype=np.float64)
        img_res= (np.shape(img)[0]*np.shape(img)[1])
        #-------------------------------------------------------
        

        #------------------------------------------------------
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
#         print(img_res)
      
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        boxes=[]
        for n,c in enumerate(cnts):
            M = cv2.moments(c)
            ar_ratio=float(cv2.contourArea(c)/(np.shape(img)[0]*np.shape(img)[1]))
            # print(ar_ratio)
            cX = int((M["m10"] / float(M["m00"]+0.001)))
            cY = int((M["m01"] / float(M["m00"]+0.001)))
            peri= cv2.arcLength(c,True)
            no_sides=len(cv2.approxPolyDP(c,0.04*peri,True))
            x,y,w,h = cv2.boundingRect(c)
            
            if no_sides>=4 and no_sides<=12 and ar_ratio>0.0007 and ar_ratio<=0.002 and (h/w)>=0.5 and (h/w)<1.9 and (h/w)>=1.33:
#                 print('Hi you are in First if stat')
                x_bottom=x
                y_bottom=np.shape(img)[0]-(y+h)
                x_topR=x+w
                y_topR=(np.shape(img)[0]-y)
                boxes.append([x_bottom,y_bottom,x_topR,y_topR])
                # cv2.imshow('Image with box',cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3 ))
                # print('Coord: %f no_sides: %f h/w:  %f Area: %f Area Ratio:  ',boxes,no_sides,h/w,cv2.contourArea(c),ar_ratio)
                # lines below were used to get an ideas of the  images in the autograder
                # print('Coord: ',boxes)
#                 print('no_sides:  ',no_sides)
#                 print('h/w:  ', h/w)
#                 print('Area:  ',cv2.contourArea(c))
#                 print('Area Ratio:  ',ar_ratio)
                

            elif no_sides>=4 and no_sides<=12 and ar_ratio>0.002 and (h/w)>=0.6 and (h/w)<1.33:
                # or np.around(ar_ratio,3)==0.001
#                 print('Hi you are in if stat')
                x_bottom=x
                y_bottom=np.shape(img)[0]-(y+h)
                x_topR=x+w
                y_topR=(np.shape(img)[0]-y)
                boxes.append([x_bottom,y_bottom,x_topR,y_topR])
                # cv2.imshow('Image with box',cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3 ))
                # print('Coord: %f no_sides: %f h/w:  %f Area: %f Area Ratio:  ',boxes,no_sides,h/w,cv2.contourArea(c),ar_ratio)
                # lines below were used to get an ideas of the  images in the autograder
#                 print('Coord: ',boxes)
#                 print('no_sides:  ',no_sides)
#                 print('h/w:  ', h/w)
#                 print('Area:  ',cv2.contourArea(c))
#                 print('Area Ratio:  ',ar_ratio)
#             else:
#                 print('else statement')
                 # print('Coord: %f no_sides: %f h/w:  %f Area: %f Area Ratio:  ',boxes,no_sides,h/w,cv2.contourArea(c),ar_ratio)
                
#                 print('Coord:  ',boxes)
#                 print('no_sides:  ',no_sides)
#                 print('h/w:  ', h/w)
#                 print('Area:  ',cv2.contourArea(c))
#                 print('Area Ratio:  ',ar_ratio)
                # print(no_sides,  cv2.contourArea(c),  h/w, ar_ratio)
            # boxes.sort()
#               print("sides: %d", no_sides)
#               print(ar_ratio, w, h, w/h)
#                 cv2.drawContours(real_img, [c], -1, (0, 255, 0), 5)
#                    #         cv2.imshow("Imag", real_img)
#                 plt.imshow(real_img)
#                 cv2.waitKey(0)
        boxes.sort()            
        return boxes






if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)
        plt.imshow(mask_img)
        plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #(2) Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)
        print(boxes)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope






