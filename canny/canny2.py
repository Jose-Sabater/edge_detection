
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import time
start=time.time()
os.chdir("D:\\Python\\General\\03_Projects\\01Practice\\Excercises\\Northvolt\\canny")
# Open the image
img = np.array(Image.open('deviation.jpg')).astype(np.uint8)
# Apply Canny
edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

plt.figure()
plt.title('canny')
plt.imsave('canny_result_deviation.png', edges, cmap='gray', format='png')
end=time.time()
print(end-start)
plt.imshow(edges, cmap='gray')
plt.show()