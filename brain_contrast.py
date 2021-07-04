import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

img1 = cv2.imread('mri_median.pgm')
img = cv2.imread('mri_t2.pgm')
ta = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
tb = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(tb)
res = np.hstack((ta,tb,equ)) #stacking images side-by-side
cv2.imwrite('res2.pgm',res)
cv2.imwrite('hist.pgm',equ)   
