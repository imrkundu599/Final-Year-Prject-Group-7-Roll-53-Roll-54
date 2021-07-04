import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

img1 = cv2.imread('mri_median.pgm')
img2 = cv2.imread('hist.pgm')
img = cv2.imread('mri_t2.pgm')
img3 = cv2.imread('preprocessed_Y7.pgm')
ta = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
tb = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)
tc = cv2.cvtColor (img2, cv2.COLOR_BGR2GRAY)
td = cv2.cvtColor (img3, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(tb, 100, 255, cv2.THRESH_BINARY) 
ret, thresh6 = cv2.threshold(td, 100, 255, cv2.THRESH_BINARY) 

ret, thresh2 = cv2.threshold(tb, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh3 = cv2.threshold(tb, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh4 = cv2.threshold(tb, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh5 = cv2.threshold(tb, 120, 255, cv2.THRESH_TOZERO_INV)



cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold tumor', thresh6) 
#cv2.imshow('Binary Threshold Inverted', thresh2) 
#cv2.imshow('Truncated Threshold', thresh3) 
#cv2.imshow('Set to 0', thresh4) 
#cv2.imshow('Set to 0 Inverted', thresh5) 
#cv2.imshow('Binary Threshold', thresh6)

#cv2.imwrite('seg.pgm',thresh1)
#cv2.imwrite('seg_tumor.pgm', thresh6)

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  