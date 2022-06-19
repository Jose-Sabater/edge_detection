import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
start=time.time()
os.chdir("D:\\Python\\General\\03_Projects\\01Practice\\Excercises\\Northvolt\\sobel")
# Open the image
img = np.array(Image.open('image.jpeg')).astype(np.uint8)
# Apply gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply gaussian blur
blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

# Positive Laplacian Operator
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)

plt.figure()
plt.title('laplacian')
plt.imsave('laplacian-result.png', laplacian, cmap='gray', format='png')
end=time.time()
print(end-start)
plt.imshow(laplacian, cmap='gray')
plt.show()