import cv2
import numpy as np

img = cv2.imread("2.png", cv2.IMREAD_UNCHANGED)
print(img.shape)
print(np.unique(img[...,0] - img[...,1]))
print(np.unique(img[...,1] - img[...,2]))
