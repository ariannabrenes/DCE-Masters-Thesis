import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim
from scipy.io import savemat

# data = sio.loadmat('Breast Sim Pipeline/DRO_data/Malignant/Run1_BC.mat') # experiment
g = h5py.File('BC_case_2_run.h5', 'r')

# f = data['IMG']
# f = np.transpose(f, (2, 0, 1))
# f = abs(f[:, None, ...])

param = g['param'][:]
output = g['output'][:]

param_c = param[..., 0] + 1j * param[..., 1]
g.close()


# Compute SSIM
#data = sio.loadmat('Breast Sim Pipeline/Patlak_Results/Fully sampled/Malignant/Patlak_BC1.h5')

#breast_img = data['IMG']
# breast_img = np.transpose(breast_img, (2, 0, 1))
# breast_img = abs(breast_img[:, None, ...])

# read_in_img = data['grog_img']
# grog_img = np.transpose(abs(read_in_img), (2, 3, 0, 1))
# coil_combined_img = np.sqrt(np.sum(grog_img**2, axis=1))
# breast_img = coil_combined_img[:, None, ...]
#
# parmap = data['parMap']
# parmap_r = parmap.transpose(2, 0, 1)
# vp_dro = parmap_r[1, ...]
# fp_dro = parmap_r[2, ...]
# ps_dro = parmap_r[3, ...]
# ktrans_dro = fp_dro * (ps_dro/(fp_dro + ps_dro))

ssim_values = []
# Loop through each pair of images
for i in range(output.shape[0]):
    # Extract the individual images
    img1 = f[i, 0, :, :]
    img2 = output[i, 0, :, :]

    # Compute SSIM between the two images
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    ssim_values.append(ssim_value)

# Calculate the mean SSIM
mean_ssim = np.mean(ssim_values)
print("Mean SSIM:", mean_ssim)

plt.figure()
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 15))
axes = axes.flatten()
for i in range(22):
    ax = axes[i]
    im = ax.imshow(f[i, 0, :, :], cmap='gray', aspect='auto')
    ax.set_title(f'Frame {i+1}')
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.axis('off')

# Hide the empty subplots
for i in range(22, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('DRO Signal Plots')
plt.show()

plt.figure()
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 15))
axes = axes.flatten()
for i in range(22):
    ax = axes[i]
    im = ax.imshow(output[i, 0, :, :], cmap='gray', aspect='auto')
    ax.set_title(f'Frame {i+1}')
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.axis('off')

# Hide the empty subplots
for i in range(22, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('Pipeline Output Signal Plots')
plt.show()


# # Visualize Ktrans and Vp
# k_trans = np.squeeze(param[0, ...])
# v_p = np.squeeze(param[1, ...])
#
# k_trans2 = np.squeeze(param2[0, ...])
# v_p2 = np.squeeze(param2[1, ...])
#
# print('Max of Ktrans: ', np.max(abs(k_trans)))
# print('Max of Vp: ', np.max(abs(v_p)))
#
# print('Avg of Ktrans: ', np.mean(abs(k_trans)))
# print('Avg of Vp: ', np.mean(abs(v_p)))
#
# f, axs = plt.subplots(2, 2, figsize=(12, 10))
#
# f.suptitle('DCE Breast FAU Estimation')
# # change the color map to match MATLAB
# im1 = axs[0, 0].imshow(abs(k_trans), cmap='gray', vmax = 0.010)
# axs[0, 0].set_title('Fully Samp Rad ktrans')
# cbar1 = f.colorbar(im1, ax=axs[0, 0])
# axs[0,0].axis('off')
#
# im2 = axs[0, 1].imshow(abs(v_p), cmap='gray')
# axs[0, 1].set_title('Full Samp Rad vp')
# cbar2 = f.colorbar(im2, ax=axs[0, 1])
# axs[0, 1].axis('off')
#
# # Plot ground truth Ktrans map
# im3 = axs[1, 0].imshow(abs(k_trans2), cmap='gray')
# axs[1, 0].set_title('Fully Sampled Cart Ktrans')
# f.colorbar(im3, ax=axs[1, 0])
# axs[1, 0].axis('off')
#
# # Plot ground truth vp map
# im4 = axs[1, 1].imshow(abs(v_p2), cmap='gray')
# axs[1, 1].set_title('Fully Samp Cart vp')
# f.colorbar(im4, ax=axs[1, 1])
# axs[1, 1].axis('off')
#
# plt.show()