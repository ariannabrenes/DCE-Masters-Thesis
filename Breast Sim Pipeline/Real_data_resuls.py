import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('BC_case_1_coils_run.h5', 'r')

param = f['param'][:]

param_c = param[..., 0] + 1j * param[..., 1]
f.close()

k_trans = np.squeeze(param_c[0, ...])
v_p = np.squeeze(param_c[1, ...])

f, ax = plt.subplots(1, 2, figsize=(10, 6))

f.suptitle('DCE Breast FAU Estimation')
# change the color map to match MATLAB
im1 = ax[0].imshow(abs(k_trans), cmap='gray')
ax[0].set_title('k_trans')
cbar1 = f.colorbar(im1, ax= ax[0])

im2 = ax[1].imshow(abs(v_p), cmap='gray')
ax[1].set_title('v_p')
cbar2 = f.colorbar(im2, ax = ax[1])

plt.show()