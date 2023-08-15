from auto_mi.tasks import TrojanMNISTTask, VAL
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

t = TrojanMNISTTask()
e = t.get_dataset(0)

im = e.trojan_image
plt.imshow(im.squeeze(0), cmap='gray')  # Use cmap='gray' for grayscale images
plt.axis('off')  # Hide axes
plt.show()

e = t.get_dataset(0, type=VAL)

im = e.trojan_image
plt.imshow(im.squeeze(0), cmap='gray')  # Use cmap='gray' for grayscale images
plt.axis('off')  # Hide axes
plt.show()