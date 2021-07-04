import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance


#img4 = cv2.imread('seg_kmean.pgm')
img4 = cv2.imread('seg_kmean_tumor.pgm')
te = cv2.cvtColor (img4, cv2.COLOR_BGR2GRAY)

img_sobelx1 = cv2.Sobel(te,cv2.CV_8U,1,0,ksize=5)
img_sobely1 = cv2.Sobel(te,cv2.CV_8U,0,1,ksize=5)
img_sobel1 = img_sobelx1 + img_sobely1

canny_edges = cv2.Canny(te,100,200)
# cv2.imshow('edge_detected',canny_edges)
# cv2.imwrite('canny_on_segmented_kmean.pgm',canny_edges)
cv2.imshow('sobel_on_segmented_kmean',img_sobel1)
cv2.imshow('canny_on_segmented_kmean',canny_edges)
#cv2.imwrite('canny_on_segmented_kmean_tumor.pgm',canny_edges)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
