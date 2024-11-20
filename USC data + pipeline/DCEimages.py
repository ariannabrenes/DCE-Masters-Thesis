import h5py
import matplotlib.pyplot as plt
import numpy as np

# The 50 anatomical images, first 8 are baseline
f = h5py.File('DCE.h5', 'r')
print(f.keys())

img = f['img'][:]
f.close()

print(img.shape)

vmax = abs(img).max()*0.5
img_display = np.swapaxes(img, -2, -1)


f, ax = plt.subplots(5, 10, figsize=(16, 12))

for m in range(5):
  for n in range(10):
    ind = m*10 + n
    ax[m, n].imshow(abs(np.squeeze(img_display[ind, ...])), vmin=0, vmax=vmax, cmap='gray', interpolation='None')
    ax[m, n].set_axis_off()

f.tight_layout()
plt.show()