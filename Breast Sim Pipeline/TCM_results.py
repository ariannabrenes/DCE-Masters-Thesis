import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage

f = h5py.File('SNB_run2.h5', 'r') # experiment
param = f['param'][:]

f.close()

print(param.shape)

ve = np.squeeze(param[0, ...])
vp = np.squeeze(param[1, ...])
fp = np.squeeze(param[2, ...])
PS = np.squeeze(param[3, ...])

# savemat("FAU_Gt_Kt_est.mat", {'gt_kt':k_trans})
# savemat("FAU_Gt_Vp_est.mat", {'gt_vp':v_p})

print('Max of ve: ', np.max(ve))
print('Max of vp: ', np.max(vp))
print('Max of Fp: ', np.max(fp))
print('Max of PS: ', np.max(PS))


f1, ax = plt.subplots(1, 4, figsize=(10, 6))

# f.suptitle('DCE Breast FAU Estimation')
#
# # change the color map to match MATLAB
# im1 = ax[0].imshow(abs(k_trans), cmap='gray')
# ax[0].set_title('k_trans')
# cbar1 = f.colorbar(im1)
#
# im2 = ax[1].imshow(abs(v_p), cmap='gray')
# ax[1].set_title('v_p')
# cbar2 = f.colorbar(im2)

f1.suptitle('DCE Breast FAU Estimation')
# change the color map to match MATLAB
im1 = ax[0].imshow(abs(ve), cmap='gray')
ax[0].set_title('ve')
cbar1 = f1.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(abs(vp), cmap='gray')
ax[1].set_title('v_p')
cbar2 = f1.colorbar(im2, ax=ax[1])

im3 = ax[2].imshow(fp, cmap='gray')
ax[2].set_title('fp')
cbar3 = f1.colorbar(im3, ax=ax[2])

im4 = ax[3].imshow(PS, cmap='gray')
ax[3].set_title('PS')
cbar4 = f1.colorbar(im4, ax=ax[3])

for m in range(4):
  ax[m].set_axis_off()

plt.show()

# set up another figure and show the difference images
f2, ax = plt.subplots(1, 4, figsize=(10, 6))

# compute difference images
dro_data = sio.loadmat('DRO Runs/DRO_RUN_1.mat')
parmap = dro_data['parMap']

dro_ve = parmap[..., 0]
dro_vp = parmap[..., 1]
dro_fp = parmap[..., 2]
dro_ps = parmap[..., 3]

# adjust data ranges here
ve_diff = skimage.metrics.structural_similarity(ve, dro_ve, data_range=...)
vp_diff = skimage.metrics.structural_similarity(vp, dro_vp, data_range=255)
fp_diff = skimage.metrics.structural_similarity(fp, dro_fp, data_range=255)
ps_diff = skimage.metrics.structural_similarity(PS, dro_ps, data_range=255)

print('SSIM VE: ', ve_diff)
print('SSIM VP: ', vp_diff)
print('SSIM FP: ', fp_diff)
print('SSIM FP: ', ps_diff)
