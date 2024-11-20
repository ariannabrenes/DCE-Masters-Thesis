import h5py
import numpy as np
import matplotlib.pyplot as plt

gt = h5py.File('GT_TKmaps.h5', 'r')

gt_param = gt['param'][:]
gt.close()

print(gt_param.shape)

gt_param = np.swapaxes(gt_param, -2, -1)

k_trans_gt = np.squeeze(gt_param[0, ...])
v_p_gt = np.squeeze(gt_param[1, ...])

print('Ktrans max: ', np.max(k_trans_gt))
print('Vp max: ', np.max(v_p_gt))

print('Ktrans avg: ', np.mean(k_trans_gt))
print('Vp avg: ', np.mean(v_p_gt))

f, ax = plt.subplots(1, 2, figsize=(10, 6))

# change the color map to match MATLAB
im1 = ax[0].imshow(abs(k_trans_gt), cmap='gray', vmin=0, vmax=0.5)
ax[0].set_title('Ground Truth Ktrans')
cbar1 = f.colorbar(im1, ax= ax[0])

im2 = ax[1].imshow(abs(v_p_gt), cmap='gray', vmin=0, vmax=0.5)
ax[1].set_title('Ground Truth Vp')
cbar2 = f.colorbar(im2, ax= ax[1])

f.subplots_adjust(top=0.85)

plt.show()