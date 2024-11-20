import numpy as np
import numpy.typing as npt
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.morphology import disk


def _fun_ktrans(x):
    """Calculate ktrans value.

    Args:
        x: Input.

    Returns:
        Ktrans value.
    """
    return x[2] * (1 - np.exp(-x[3] / x[2]))


def _generate_random_params(p0, ktrans_range):
    """Function to generate random parameter values within specified ranges

    Args:
        p0: p0.
        ktrans_range: ktrans range.

    Returns:
        Random parameter.
    """
    while True:
        params = p0[:, 0] + (p0[:, 1] - p0[:, 0]) * np.random.rand(4)
        par_ktrans = _fun_ktrans(params)
        if ktrans_range[0] <= par_ktrans <= ktrans_range[1]:
            break
    return params


def _add_field(
    struct_arr: npt.NDArray, new_field_data: npt.NDArray, field_descr: list
) -> npt.NDArray:
    """Adds a field with the given specification to a structured numpy array.

    Args:
        struct_arr: Structured numpy.
        new_field_data: Value for the new field.
        field_descr: Description for the new field.

    Raises:
        ValueError: Raised if given array is not a structured numpy array.

    Returns:
        Structured numpy array containing the new field.
    """
    if struct_arr.dtype.fields is None:
        raise ValueError('Input array must be a structured numpy array')
    b = np.empty(struct_arr.shape, dtype=struct_arr.dtype.descr + field_descr)
    for name in struct_arr.dtype.names:
        b[name] = struct_arr[name]
    b[field_descr[0][0]] = new_field_data
    return b


def gen_simIMG2(data, idx=None, B1=None, parMap=None) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    str,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """Generate synthetic DCE MR data.

    Args:
        data: Breast Data of all subjects.
        idx: Subject index. Defaults to None.
        B1: B1. Defaults to None.
        parMap: parMap. Defaults to None.

    Returns:
        Simulated image (simImg), simulated sensitivity map (smap),
        parameter map (parMap), temp, ID, Arterial Input Function (aif),
        T10, aifci_1s, concentration-time curves (cts)
    """
    # If no subject index provided, select a random one
    if idx is None:
        idx = np.random.randint(0, len(data))

    par_var = 0.1
    par_var_t = 0.2
    t = np.linspace(0, 150, 22)  # Time points
    mask = data[idx]['mask'].item()  # Mask for the subject
    aif = data[idx]['AIF'].item()  # Arterial Input Function
    S0 = data[idx]['S0'].item()  # Initial signal intensity

    # If B1 field not provided, assume uniform
    if B1 is None:
        B1 = np.ones_like(S0)

    # Subject ID
    ID = str(data[idx]['ID'])
    print(ID)

    # Sensitivity map
    smap = np.real(data[idx]['smap'].item()).astype(np.float64)

    # Add missing fields to the mask if they don't exist
    if 'heart' not in mask.dtype.names:
        mask['heart'] = np.zeros_like(S0, dtype=bool)
    if 'liver' not in mask.dtype.names:
        mask['liver'] = np.zeros_like(S0, dtype=bool)
    if 'skin' not in mask.dtype.names:
        mask['skin'] = np.zeros_like(S0, dtype=bool)
    if 'muscle' not in mask.dtype.names:
        mask['muscle'] = np.zeros_like(S0, dtype=bool)
    if 'benign' not in mask.dtype.names:
        mask['benign'] = np.zeros_like(S0, dtype=bool)
    if 'malignant' not in mask.dtype.names:
        mask['malignant'] = np.zeros_like(S0, dtype=bool)
    if 'heart_blood' not in mask.dtype.names:
        mask = _add_field(mask, mask['heart'], [('heart_blood', '|O')])

    # Interpolate AIF to have samples at every second
    nbase = np.where((aif < 0.15) & (t < 300))[0][-1]
    t_1s = np.arange(0, t[-1] + 1)
    aifci = pchip_interpolate(t, aif, t_1s)

    nx, ny = S0.shape
    mask_inner = mask['heart'].item().astype(bool) | mask[
        'liver'
    ].item().astype(bool)
    selem = disk(4)
    mask_inner = binary_dilation(mask_inner, selem)

    # T1 relaxation times for different tissues
    T1 = {
        'glandular': 1.324,
        'malignant': 1.5,
        'benign': 1.4,
        'liver': 0.81,
        'heart': 0.81,
        'muscle': 1.41,
        'skin': 0.85,
        'vascular': 1.93,
    }

    # Assign T1 values to the corresponding tissue regions in the mask
    T10 = np.zeros((nx, ny))
    for tissue, value in T1.items():
        T10[mask[tissue].item().astype(bool)] = value
    T10[T10 == 0] = 1
    temp = T10.copy()
    temp = gaussian_filter(temp, sigma=10)
    T10[mask_inner] = temp[mask_inner]

    # If parameter map not provided, use default values
    if parMap is None:
        p0 = {
            'glandular': np.array(
                [
                    [0, 0],
                    [0.010, 0.077],
                    [0.1 / 60, 0.115 / 60],
                    [0.01 / 60, 0.043 / 60],
                ]
            ),
            'malignant': np.array(
                [
                    [0.101, 0.3],
                    [0.131, 0.256],
                    [0.259 / 60, 1.032 / 60],
                    [0.0434 / 60, 1.98 / 60],
                ]
            ),
            'benign': np.array(
                [
                    [0.141, 0.3],
                    [0.011, 0.190],
                    [0.116 / 60, 0.228 / 60],
                    [0.05 / 60, 1.056 / 60],
                ]
            ),
            'muscle': np.array(
                [
                    [0, 0],
                    [0.010, 0.101],
                    [0.1 / 60, 0.118 / 60],
                    [0.011 / 60, 0.069 / 60],
                ]
            ),
            'skin': np.array(
                [
                    [0, 0],
                    [0.039, 0.125],
                    [0.1 / 60, 0.151 / 60],
                    [0.01 / 60, 0.019 / 60],
                ]
            ),
            'liver': np.array(
                [
                    [0.1, 0.5],
                    [0.353, 0.5],
                    [0.433 / 60, 1.227 / 60],
                    [1.990 / 60, 2 / 60],
                ]
            ),
            'heart': np.array(
                [
                    [0.148, 0.300],
                    [0.214, 0.373],
                    [2 / 60, 2 / 60],
                    [0.404 / 60, 1.224 / 60],
                ]
            ),
            'vascular': np.array(
                [[0, 0], [0.3, 0.3], [2 / 60, 2 / 60], [0 / 60, 0 / 60]]
            ),
        }

        gland_ktrans = np.array([0.01, 0.0352]) / 60
        malig_ktrans = np.array([0.0412, 0.385]) / 60
        benign_ktrans = np.array([0.0453, 0.143]) / 60
        muscle_ktrans = np.array([0.011, 0.05]) / 60
        skin_ktrans = np.array([0.009, 0.017]) / 60
        liver_ktrans = np.array([0.412, 0.979]) / 60
        heart_ktrans = np.array([0.365, 0.810]) / 60

        # Generate parameters for glandular tissue
        p0_glandular = _generate_random_params(p0['glandular'], gland_ktrans)

        # Generate parameters for malignant tissue
        p0_malignant = _generate_random_params(p0['malignant'], malig_ktrans)

        # Generate parameters for benign tissue
        p0_benign = _generate_random_params(p0['benign'], benign_ktrans)

        # Generate parameters for muscle tissue
        p0_muscle = _generate_random_params(p0['muscle'], muscle_ktrans)

        # Generate parameters for skin tissue
        p0_skin = _generate_random_params(p0['skin'], skin_ktrans)

        # Generate parameters for heart tissue
        p0_heart = _generate_random_params(p0['heart'], heart_ktrans)

        # Generate parameters for liver tissue
        p0_liver = _generate_random_params(p0['liver'], liver_ktrans)

        p0_vascular = p0['vascular'][:, 0] + (
            p0['vascular'][:, 1] - p0['vascular'][:, 0]
        ) * np.random.rand(4)

        # Initialize parMap as a zeros matrix of size (nx, ny, 4)
        parMap = np.zeros((nx, ny, 4))

        for i in range(4):
            # Initialize a temporary matrix filled with zeros
            temp = np.zeros((nx, ny))

            # Generate random maps for different tissue types and assign them to the temp matrix
            randMap = p0_glandular[i] * (
                (1 - par_var) + (par_var * 2) * np.random.rand(nx, ny)
            )
            temp[mask['glandular'].item().astype(bool)] = randMap[
                mask['glandular'].item().astype(bool)
            ]

            randMap = p0_malignant[i] * (
                (1 - par_var_t) + (par_var_t * 2) * np.random.rand(nx, ny)
            )
            temp[mask['malignant'].item().astype(bool)] = randMap[
                mask['malignant'].item().astype(bool)
            ]

            randMap = p0_benign[i] * (
                (1 - par_var_t) + (par_var_t * 2) * np.random.rand(nx, ny)
            )
            temp[mask['benign'].item().astype(bool)] = randMap[
                mask['benign'].item().astype(bool)
            ]

            randMap = p0_liver[i] * (
                (1 - par_var) + (par_var * 2) * np.random.rand(nx, ny)
            )
            temp[mask['liver'].item().astype(bool)] = randMap[
                mask['liver'].item().astype(bool)
            ]

            randMap = p0_muscle[i] * (
                (1 - par_var) + (par_var * 2) * np.random.rand(nx, ny)
            )
            temp[mask['muscle'].item().astype(bool)] = randMap[
                mask['muscle'].item().astype(bool)
            ]

            randMap = p0_skin[i] * (
                (1 - par_var) + (par_var * 2) * np.random.rand(nx, ny)
            )
            temp[mask['skin'].item().astype(bool)] = randMap[
                mask['skin'].item().astype(bool)
            ]

            randMap = p0_vascular[i] * np.ones((nx, ny))
            temp[mask['vascular'].item().astype(bool)] = randMap[
                mask['vascular'].item().astype(bool)
            ]

            randMap = p0_heart[i] * (
                (1 - par_var) + (par_var * 2) * np.random.rand(nx, ny)
            )
            temp[mask['heart_blood'].item().astype(bool)] = randMap[
                mask['heart_blood'].item().astype(bool)
            ]

            # Assign the temporary matrix to the corresponding slice of parMap
            parMap[:, :, i] = temp

        # Initialize aifci_1s as a zeros array of size (nx, ny, len(t_1s))
        aifci_1s = np.zeros((nx, ny, len(t_1s)))

        # Iterate over mask.liver
        rIdx, cIdx = np.where(mask['liver'].item() == 1)  # 10s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 10, np.arange(1, len(t_1s) - 9)))
            ]

        # Iterate over mask.glandular
        rIdx, cIdx = np.where(mask['glandular'].item() == 1)  # 15s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 15, np.arange(1, len(t_1s) - 14)))
            ]

        # Iterate over mask.vascular or mask.heart_blood
        rIdx, cIdx = np.where(
            (mask['vascular'].item() | mask['heart_blood'].item()) == 1
        )  # 0s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci

        # Iterate over mask.malignant
        rIdx, cIdx = np.where(mask['malignant'].item() == 1)  # 3s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 3, np.arange(1, len(t_1s) - 2)))
            ]

        # Iterate over mask.benign
        rIdx, cIdx = np.where(mask['benign'].item() == 1)  # 8s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 8, np.arange(1, len(t_1s) - 7)))
            ]

        # Iterate over mask.muscle
        rIdx, cIdx = np.where(mask['muscle'].item() == 1)  # 7s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 7, np.arange(1, len(t_1s) - 6)))
            ]

        # Iterate over mask.skin
        rIdx, cIdx = np.where(mask['skin'].item() == 1)  # 10s delay
        for i in range(len(rIdx)):
            aifci_1s[rIdx[i], cIdx[i], :] = aifci[
                np.concatenate(([1] * 5, np.arange(1, len(t_1s) - 4)))
            ]

        # Initialize ti - steps of every 0.1 seconds
        ti = np.arange(0, t[-1] + 0.1, 0.1)

        # Initialize logIdx as a zeros array of length len(ti)
        logIdx = np.zeros(len(ti))

        # Populate logIdx based on t values
        start_idx = 0
        for i in range(len(t)):
            for j in range(start_idx, len(ti)):
                if t[i] <= ti[j]:
                    logIdx[j] = 1
                    start_idx = j
                    break

        # Convert logIdx to a logical array
        logIdx = logIdx.astype(bool)

        # Initialize aifci_Map as a zeros array of size (nx, ny, len(ti))
        aifci_Map = np.zeros((nx, ny, len(ti)))

        # Iterate over parMap[:,:,1] > 0 - where vp is greater than 0
        rIdx, cIdx = np.where(parMap[:, :, 1] > 0)
        for i in range(len(rIdx)):
            temp_aif = np.squeeze(aifci_1s[rIdx[i], cIdx[i], :])
            aifci_Map[rIdx[i], cIdx[i], :] = pchip_interpolate(
                t_1s, temp_aif, ti
            )

        # Adjust parMap values and apply Gaussian filter
        parMap[parMap == 0] = 1e-8
        for i in range(4):
            par_temp = gaussian_filter(parMap[:, :, i], sigma=1)
            temp = np.copy(parMap[:, :, i])
            temp[~mask_inner] = 0
            temp = gaussian_filter(temp, sigma=20)
            par_temp[mask_inner] = temp[mask_inner]
            parMap[:, :, i] = par_temp

        # Extract parameters from parMap
        ve = parMap[:, :, 0]
        vp = parMap[:, :, 1]
        fp = parMap[:, :, 2]
        ktrans = parMap[:, :, 3]

        # Calculate concentration-time curves (cts)
        Ce = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))
        Cp = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))

        for i in range(1, aifci_Map.shape[2]):
            dt = ti[i] - ti[i - 1]
            dcp = (
                fp * aifci_Map[:, :, i - 1]
                - (fp + ktrans) * Cp[:, :, i - 1]
                + ktrans * Ce[:, :, i - 1]
            )
            dce = ktrans * Cp[:, :, i - 1] - ktrans * Ce[:, :, i - 1]
            Cp[:, :, i] = Cp[:, :, i - 1] + dcp * dt / vp
            Ce[:, :, i] = Ce[:, :, i - 1] + dce * dt / ve

        cts = Cp * np.tile(
            vp[:, :, np.newaxis], (1, 1, len(ti))
        ) + Ce * np.tile(ve[:, :, np.newaxis], (1, 1, len(ti)))
        cts = cts[:, :, logIdx]

        # Optionally handle NaN values
        # nan_mask = np.isnan(cts)
        # cts[nan_mask] = 0

        # Assign cts to cts_tcm
        cts_tcm = np.copy(cts)
        conc_tcm = cts_tcm
#----------------------------------------------------------------------------------------------------------------------

        # Assigning parameters ve, vp, fp, and ktrans from parMap
        ve = 1
        vp = parMap[:, :, 1]
        fp = parMap[:, :, 2]
        ktrans = parMap[:, :, 3]

        Ce = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))
        Cp = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))

        for i in range(1, aifci_Map.shape[2]):
            dt = ti[i] - ti[i - 1]
            dcp = fp * aifci_Map[:, :, i - 1] - (fp + ktrans) * Cp[:, :, i - 1]
            dce = ktrans * Cp[:, :, i - 1]
            Cp[:, :, i] = Cp[:, :, i - 1] + dcp * dt / vp
            Ce[:, :, i] = Ce[:, :, i - 1] + dce * dt / ve

        cts = Cp * np.tile(
            vp[:, :, np.newaxis], (1, 1, len(ti))
        ) + Ce * np.tile(ve, (parMap.shape[0], parMap.shape[1], len(ti)))
        cts = cts[:, :, logIdx]

        # # Handling NaN values
        # nan_mask = np.isnan(cts)
        # cts[nan_mask] = 0

        # Assign cts to cts_epm
        cts_epm = np.copy(cts)

        ve = 1
        vp = parMap[:, :, 1]
        ktrans = parMap[:, :, 3]

        Ce = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))
        Cp = np.zeros((parMap.shape[0], parMap.shape[1], aifci_Map.shape[2]))

        for i in range(1, aifci_Map.shape[2]):
            dt = ti[i] - ti[i - 1]
            dce = ktrans * aifci_Map[:, :, i - 1]
            Ce[:, :, i] = Ce[:, :, i - 1] + dce * dt / ve

        cts = aifci_Map * np.tile(vp[..., np.newaxis], (1, 1, len(ti))) + Ce
        cts = cts[:, :, logIdx]

        # Handling NaN values
        nan_mask = np.isnan(cts)
        cts[nan_mask] = 0

        # Assign cts to cts_eTofts
        cts_eTofts = np.copy(cts)

        # Define mask_epm and mask_eTofts
        mask_epm = (
            mask['glandular'].item()
            | mask['skin'].item()
            | mask['muscle'].item()
        )
        mask_epm = np.tile(
            mask_epm[:, :, np.newaxis], (1, 1, cts.shape[2])
        ).astype(bool)
        mask_eTofts = mask['vascular'].item()
        mask_eTofts = np.tile(
            mask_eTofts[:, :, np.newaxis], (1, 1, cts.shape[2])
        ).astype(bool)

        # Combine cts_epm and cts_eTofts based on masks
        cts_tcm[mask_epm] = cts_epm[mask_epm]
        cts_tcm[mask_eTofts] = cts_eTofts[mask_eTofts]
        cts = cts_tcm

        # Count NaN voxels
        nan_mask = np.sum(np.isnan(cts), axis=2)
        nan_voxels = np.count_nonzero(nan_mask)
        print(f"Found {nan_voxels} NaN voxels")

        # Calculate TR, theta, and Eh
        TR = 4.87e-3
        theta = 10 * np.pi / 180
        theta = theta * B1
        theta = np.expand_dims(theta, axis=-1)
        r1 = 4.3
        T1_t = 1 / (1 / T10[:, :, np.newaxis] + r1 * cts)

        Eh = (
            (1 - np.exp(-TR / T1_t))
            * np.sin(theta)
            / (1 - (np.exp(-TR / T1_t) * np.cos(theta)))
        )
        mean_Eh = np.mean(Eh[:, :, :nbase], axis=2)
        Eh /= np.tile(mean_Eh[:, :, np.newaxis], (1, 1, Eh.shape[2]))

        # Set zero values for NaN elements
        Eh[np.isnan(Eh)] = 0

        # Calculate simImg
        simImg = Eh * np.tile(S0[:, :, np.newaxis], (1, 1, Eh.shape[2]))

        # Handling NaN values in simImg
        nan_mask = np.isnan(simImg)
        nan_voxels = np.count_nonzero(nan_mask)
        print(f"Sig-Found {nan_voxels} NaN voxels")

        # Optionally handle NaN values visualization
        # if nan_voxels > 0:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(np.abs(S0), cmap='gray')
        #     plt.hold(True)
        #     plt.imshow(nan_mask > 0, cmap='jet')
        #     plt.show()

        return simImg, smap, parMap, temp, ID, aif, T10, aifci_1s, cts, conc_tcm