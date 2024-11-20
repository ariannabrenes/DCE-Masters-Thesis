import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as stats

# Define case type and index number
case_type = 'Benign'  # Change to 'Benign' for benign cases
index = 24

# Define file paths based on case type
param_file_path = f'Patlak_Results/Fully sampled/{case_type}/Patlak_BC{index}.h5'if case_type == 'Malignant' else f'Patlak_Results/Fully sampled/{case_type}/Patlak_PT{index}.h5'
data_file_path = f'DRO_data/{case_type}/Run{index}_BC.mat' if case_type == 'Malignant' else f'DRO_data/{case_type}/Run{index}_PT.mat'
mask_file_path = f'DRO Masks/DRO_masks_newdata.mat' if case_type == 'Malignant' else 'DRO Masks/DRO_masks_data.mat'

# Load and process data based on case type
f = h5py.File(param_file_path, 'r')
param = f['param'][:]
output = f['output'][:]
f.close()

Est_ktrans = np.squeeze(param[0, ...])
Est_vp = abs(np.squeeze(param[1, ...]))

print(param.shape)

data = sio.loadmat(data_file_path)
breast_img = data['IMG']
breast_img = np.transpose(breast_img, (2, 0, 1))
breast_img = abs(breast_img[:, None, ...])

parmap = data['parMap']
parmap_r = parmap.transpose(2, 0, 1)
vp_dro = parmap_r[1, ...]
fp_dro = parmap_r[2, ...]
ps_dro = parmap_r[3, ...]
ktrans_dro = fp_dro * (ps_dro / (fp_dro + ps_dro))

mask_data = sio.loadmat(mask_file_path)
masks = mask_data['mask']

# Access the mask for the given patient
patient_mask = masks[0][index - 1]
if case_type == 'Malignant':
    case_mask_object = patient_mask['malignant']
else:
    case_mask_object = patient_mask['benign']
case_mask = case_mask_object[0, 0]

# Apply the mask
Est_ktrans_mask = Est_ktrans * case_mask
Est_vp_mask = Est_vp * case_mask
Dro_ktrans_mask = ktrans_dro * case_mask
Dro_vp_mask = vp_dro * case_mask

# Create subplots
f, axs = plt.subplots(2, 4, figsize=(16, 8))
#f.suptitle(f'DCE Breast FAU Estimation: {case_type} Masks')
f.suptitle(f'DCE Breast FAU Estimation and Lesion Masks')

# Top row (ktrans)
# Plot and add color bar for Est_ktrans
im1 = axs[0, 0].imshow(abs(Est_ktrans), cmap='gray', vmax=0.010)
axs[0, 0].set_title('Est Ktrans')
cbar1 = f.colorbar(im1, ax=axs[0, 0])
axs[0, 0].axis('off')

# Plot and add color bar for Est_ktrans_mask
im2 = axs[0, 1].imshow(Est_ktrans_mask, cmap='gray')
axs[0, 1].set_title('Est Ktrans Mask')
cbar2 = f.colorbar(im2, ax=axs[0, 1])
axs[0, 1].axis('off')

# Plot and add color bar for ktrans_dro
im3 = axs[0, 2].imshow(ktrans_dro, cmap='gray')
axs[0, 2].set_title('DRO Ktrans')
cbar3 = f.colorbar(im3, ax=axs[0, 2])
axs[0, 2].axis('off')

# Plot and add color bar for Dro_ktrans_mask
im4 = axs[0, 3].imshow(Dro_ktrans_mask, cmap='gray')
axs[0, 3].set_title('DRO Ktrans Mask')
cbar4 = f.colorbar(im4, ax=axs[0, 3])
axs[0, 3].axis('off')

# Bottom row (vp)
# Plot and add color bar for Est_vp
im5 = axs[1, 0].imshow(abs(Est_vp), cmap='gray')
axs[1, 0].set_title('Est Vp')
cbar5 = f.colorbar(im5, ax=axs[1, 0])
axs[1, 0].axis('off')

# Plot and add color bar for Est_vp_mask
im6 = axs[1, 1].imshow(Est_vp_mask, cmap='gray')
axs[1, 1].set_title('Est Vp Mask')
cbar6 = f.colorbar(im6, ax=axs[1, 1])
axs[1, 1].axis('off')

# Plot and add color bar for vp_dro
im7 = axs[1, 2].imshow(vp_dro, cmap='gray')
axs[1, 2].set_title('DRO Vp')
cbar7 = f.colorbar(im7, ax=axs[1, 2])
axs[1, 2].axis('off')

# Plot and add color bar for Dro_vp_mask
im8 = axs[1, 3].imshow(Dro_vp_mask, cmap='gray')
axs[1, 3].set_title('DRO Vp Mask')
cbar8 = f.colorbar(im8, ax=axs[1, 3])
axs[1, 3].axis('off')

plt.show()

# Extract only the mask values - no background zeros
Est_ktrans_mask_only = Est_ktrans_mask[Est_ktrans_mask != 0]
Est_vp_mask_only = Est_vp_mask[Est_vp_mask != 0]

Dro_ktrans_mask_only = Dro_ktrans_mask[Dro_ktrans_mask != 0]
Dro_vp_mask_only = Dro_vp_mask[Dro_vp_mask != 0]

# # PERFORM PAIRWISE T-TESTS FOR MASKS
# # T-test for Ktrans masks
# t_stat_ktrans, p_val_ktrans = stats.ttest_rel((Est_ktrans_mask_only.flatten()), Dro_ktrans_mask_only.flatten())
#
# # T-test for vp masks
# t_stat_vp, p_val_vp = stats.ttest_rel(Est_vp_mask_only.flatten(), Dro_vp_mask_only.flatten())
#
# print(f'Pairwise T-test results for Ktrans masks:')
# print(f'T-statistic: {t_stat_ktrans}, P-value: {p_val_ktrans}')
#
# print(f'\nPairwise T-test results for vp masks:')
# print(f'T-statistic: {t_stat_vp}, P-value: {p_val_vp}')

# Compute the mean value of each mask
mean_values = {
    'Est_ktrans_mask_mean': np.mean(Est_ktrans_mask_only),
    'Est_vp_mask_mean': np.mean(Est_vp_mask_only),
    'Dro_ktrans_mask_mean': np.mean(Dro_ktrans_mask_only),
    'Dro_vp_mask_mean': np.mean(Dro_vp_mask_only)
}

# Print the mean values
for key, value in mean_values.items():
    print(f'\n{key}: {value}')