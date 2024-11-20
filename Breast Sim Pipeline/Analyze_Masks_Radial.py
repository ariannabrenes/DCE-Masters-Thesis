import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as stats

# Define case type and index number
case_type = 'Benign'  # Change to 'Benign' for benign cases
index = 32

# In Radial Cases !
spoke_number = 13
#Rad_Est_file_path = f'Patlak_Results/Radial_FullySampled/{case_type}/Patlak_BC{index}_{spoke_number}sp.h5'if case_type == 'Malignant' else f'Patlak_Results/Radial_FullySampled/{case_type}/Patlak_PT{index}_{spoke_number}sp.h5'
Rad_Est_file_path = f'Patlak_Results/Undersampled/{case_type}/{spoke_number} spokes/Patlak_BC{index}_{spoke_number}sp.h5'if case_type == 'Malignant' else f'Patlak_Results/Undersampled/{case_type}/{spoke_number} spokes/Patlak_PT{index}_{spoke_number}sp.h5'
Est_file_path = f'Patlak_Results/Fully sampled/{case_type}/Patlak_BC{index}.h5'if case_type == 'Malignant' else f'Patlak_Results/Fully sampled/{case_type}/Patlak_PT{index}.h5'
mask_file_path = f'DRO Masks/DRO_masks_newdata.mat' if case_type == 'Malignant' else 'DRO Masks/DRO_masks_data.mat'


# Load and process data based on case type
f = h5py.File(Rad_Est_file_path, 'r')
param_f = f['param'][:]
output_f = f['output'][:]

g = h5py.File(Est_file_path, 'r')
param_g = g['param'][:]
output_g = g['output'][:]

g.close()
f.close()

Rad_Est_ktrans = np.squeeze(param_f[0, ...])
Rad_Est_vp = abs(np.squeeze(param_f[1, ...]))

Est_ktrans = np.squeeze(param_g[0, ...])
Est_vp = abs(np.squeeze(param_g[1, ...]))

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
Rad_Est_ktrans_mask = Rad_Est_ktrans * case_mask
Rad_Est_vp_mask = Rad_Est_vp * case_mask
Est_ktrans_mask = Est_ktrans * case_mask
Est_vp_mask = Est_vp * case_mask

# Create subplots
f, axs = plt.subplots(2, 4, figsize=(16, 8))
f.suptitle(f'DCE Breast FAU Estimation: {case_type} Masks')

# Top row (ktrans)
# Plot and add color bar for Est_ktrans
im1 = axs[0, 0].imshow(abs(Rad_Est_ktrans), cmap='gray', vmax=0.010)
axs[0, 0].set_title('Rad Est Ktrans')
cbar1 = f.colorbar(im1, ax=axs[0, 0])
axs[0, 0].axis('off')

# Plot and add color bar for Est_ktrans_mask
im2 = axs[0, 1].imshow(Rad_Est_ktrans_mask, cmap='gray')
axs[0, 1].set_title('Rad Est ktrans mask')
cbar2 = f.colorbar(im2, ax=axs[0, 1])
axs[0, 1].axis('off')

# Plot and add color bar for ktrans_dro
im3 = axs[0, 2].imshow(Est_ktrans, cmap='gray')
axs[0, 2].set_title('Est Ktrans')
cbar3 = f.colorbar(im3, ax=axs[0, 2])
axs[0, 2].axis('off')

# Plot and add color bar for Dro_ktrans_mask
im4 = axs[0, 3].imshow(Est_ktrans_mask, cmap='gray')
axs[0, 3].set_title('Est Ktrans mask')
cbar4 = f.colorbar(im4, ax=axs[0, 3])
axs[0, 3].axis('off')

# Bottom row (vp)
# Plot and add color bar for Est_vp
im5 = axs[1, 0].imshow(abs(Rad_Est_vp), cmap='gray')
axs[1, 0].set_title('Rad Est vp')
cbar5 = f.colorbar(im5, ax=axs[1, 0])
axs[1, 0].axis('off')

# Plot and add color bar for Est_vp_mask
im6 = axs[1, 1].imshow(Rad_Est_vp_mask, cmap='gray')
axs[1, 1].set_title('Rad Est vp mask')
cbar6 = f.colorbar(im6, ax=axs[1, 1])
axs[1, 1].axis('off')

# Plot and add color bar for vp_dro
im7 = axs[1, 2].imshow(Est_vp, cmap='gray')
axs[1, 2].set_title('Est vp')
cbar7 = f.colorbar(im7, ax=axs[1, 2])
axs[1, 2].axis('off')

# Plot and add color bar for Dro_vp_mask
im8 = axs[1, 3].imshow(Est_vp_mask, cmap='gray')
axs[1, 3].set_title('Est vp mask')
cbar8 = f.colorbar(im8, ax=axs[1, 3])
axs[1, 3].axis('off')

plt.show()

# Extract only the mask values - no background zeros
Rad_Est_ktrans_mask_only = Rad_Est_ktrans_mask[Rad_Est_ktrans_mask != 0]
Rad_Est_vp_mask_only = Rad_Est_vp_mask[Rad_Est_vp_mask != 0]

Est_ktrans_mask_only = Est_ktrans_mask[Est_ktrans_mask != 0]
Est_vp_mask_only = Est_vp_mask[Est_vp_mask != 0]

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
    'Rad_Est_ktrans_mask_mean': np.mean(Rad_Est_ktrans_mask_only),
    'Rad_Est_vp_mask_mean': np.mean(Rad_Est_vp_mask_only),
    'Est_ktrans_mask_mean': np.mean(Est_ktrans_mask_only),
    'Est_vp_mask_mean': np.mean(Est_vp_mask_only)
}

# Print the mean values
for key, value in mean_values.items():
    print(f'\n{key}: {value}')