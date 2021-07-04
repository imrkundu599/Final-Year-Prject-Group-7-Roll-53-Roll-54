import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance


#img = cv2.imread('mri_t2.pgm')
img = cv2.imread('Y7.pgm')
t = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

mri_median = cv2.medianBlur(t,5)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(mri_median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imwrite('preprocessed_Y7.pgm',mri_median)
#status = cv2.imwrite('C:/Users/lenovo pc/Desktop/finalyearproject/mri_median.pgm',mri_median)

#print("Image written to file-system : ",status)
