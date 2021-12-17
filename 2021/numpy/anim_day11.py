import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time

def get_frame():
    octopuses = np.loadtxt('day11_test.txt', encoding='utf8', converters={0: list}, dtype='int')
    radius = np.ones((3, 3))
    radius[1, 1] = 0
    while True:
        octopuses += 1
        yield octopuses
        flashes = octopuses > 9
        octopuses[flashes] = 0
        affected = convolve2d(flashes, radius, mode='same').astype(int)
        octopuses += affected
       
fig, axes = plt.subplots(5, 5, figsize=(9, 9))
frame_gen = get_frame()
for frame, ax in zip(range(25), axes.flatten()):
    img = next(frame_gen)
    img[img > 9] = 20
    ax.imshow(img, vmin=0, vmax=20)
    ax.set_axis_off()
plt.tight_layout()
