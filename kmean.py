import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

#img1 = cv2.imread('mri_median.pgm')
#tb = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)

img1 = cv2.imread('preprocessed_Y7.pgm')
tb = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)

pixel_vals = tb.reshape((-1,1)) 
  
# Convert to float type 
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running, 
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
#becomes 85% 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

# then perform k-means clustering wit h number of clusters defined as 3 
#also random centres are initally chosed for k-means clustering 
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
print(retval)
print(centers)
print(labels)
# convert data into 8-bit values 
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 

# reshape data into the original image dimensions 
segmented_image = segmented_data.reshape((tb.shape)) 

cv2.imshow('segmented',segmented_image)
#cv2.imwrite('seg_kmean.pgm',segmented_image)
#cv2.imwrite('seg_kmean_tumor.pgm',segmented_image)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  