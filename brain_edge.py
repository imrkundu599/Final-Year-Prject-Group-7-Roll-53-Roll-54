import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

img1 = cv2.imread('mri_median.pgm')
#img3 = cv2.imread('seg.pgm')
img3 = cv2.imread('seg_tumor.pgm')

tb = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)
td = cv2.cvtColor (img3, cv2.COLOR_BGR2GRAY)

canny_edges = cv2.Canny(tb,100,200)
img_sobelx = cv2.Sobel(tb,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(tb,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

canny_edges1 = cv2.Canny(td,100,200)
img_sobelx1 = cv2.Sobel(td,cv2.CV_8U,1,0,ksize=5)
img_sobely1 = cv2.Sobel(td,cv2.CV_8U,0,1,ksize=5)
img_sobel1 = img_sobelx1 + img_sobely1

e = np.hstack((tb,canny_edges,img_sobel)) #stacking images side-by-side
#cv2.imshow('edge-detection', e)
#cv2.imwrite('canny_on_median_filtered.pgm',canny_edges)
#cv2.imwrite('sobel_on_median_filtered.pgm',img_sobel)

#cv2.imwrite('canny_on_segmented.pgm',canny_edges1)
#cv2.imwrite('sobel_on_segemented.pgm',img_sobel1)
cv2.imshow('canny_on_segmented_thresold',canny_edges1)
#cv2.imwrite('canny_on_segemented_tumor.pgm',canny_edges1)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  