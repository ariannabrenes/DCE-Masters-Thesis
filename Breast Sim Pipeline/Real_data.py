import h5py
import os
import torch
import torch.fft as fft
from scipy.interpolate import pchip_interpolate
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dce import DCE
import dce



# from dce import DCE
def check_for_nan(x, name, index):
    if torch.isnan(x).any():
        print(f"Nan values found in {name} at index {index}")


def check_for_inf(x, name, index):
    if torch.isinf(x).any():
        print(f"Inf values found in {name} at index {index}")


# TEST - ksp to image space , matching jo's code
def fft2c(x, axes=(-2, -1), norm='ortho'):
    x = torch.fft.fftshift(x, dim=axes)
    x = fft.ifft2(x, dim=axes, norm=norm)

    # center the kspace
    x = torch.fft.fftshift(x, dim=axes)
    return x


# TEST - image to kspace
def ifft2c(x, axes=(-2, -1), norm='ortho'):
    x = torch.fft.fftshift(x, dim=axes)
    x = fft.fft2(x, dim=axes, norm=norm)

    # center the kspace
    x = torch.fft.fftshift(x, dim=axes)
    return x


class DCE(nn.Module):
    def __init__(self,
                 ishape,
                 sample_time,
                 sig_baseline,
                 Cp,
                 R1,
                 csm,
                 M0=5.0,
                 R1CA=4.30,
                 FA=10.,
                 TR=4.87E-3,
                 x_iscomplex=True,
                 undersampled = None,
                 device=torch.device('cpu')):
        super(DCE, self).__init__()

        if x_iscomplex:
            self.ishape = list(ishape[:-1])
        else:
            self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32, device=device)
        self.sig_baseline = sig_baseline

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32, device=device)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32, device=device)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32, device=device)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32, device=device)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32, device=device)

        self.FA_radian = self.FA * np.pi / 180.
        E1 = torch.exp(-self.TR * self.R1)

        topM0 = self.sig_baseline * (1 - torch.cos(self.FA_radian) * E1)
        bottomM0 = torch.sin(self.FA_radian) * (1 - E1)
        self.M0 = topM0 / bottomM0
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        # Cp = dce.arterial_input_function(sample_time)
        # self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)

        # # read in breast sim aif, resize for shape [numpoints, ]
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)
        self.Cp = self.Cp.transpose(0, 1).view(-1)
        self.csm = torch.tensor(csm, dtype=torch.complex64, device=device)
        self.undersampled = undersampled

        self.device = device

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def matmul_complex(self, torch1, t2):
        return torch.view_as_complex(
            torch.stack((torch1.real @ t2.real - torch1.imag @ t2.imag, torch1.real @ t2.imag + torch1.imag @ t2.real),
                        dim=2))

    # TEST - ksp to image space , matching jo's code
    def fft2c(self, x, axes=(-2, -1), norm='ortho'):
        x = torch.fft.fftshift(x, dim=axes)
        x = fft.ifft2(x, dim=axes, norm=norm)

        # center the kspace
        x = torch.fft.fftshift(x, dim=axes)
        return x

    # TEST - image to kspace
    def ifft2c(self, x, axes=(-2, -1), norm='ortho'):
        x = torch.fft.fftshift(x, dim=axes)
        x = fft.fft2(x, dim=axes, norm=norm)

        # center the kspace
        x = torch.fft.fftshift(x, dim=axes)
        return x

    def convolve(self, x, y):
        # Using PyTorch's conv1d with appropriate padding and kernel size
        return torch.conv1d(x.unsqueeze(0), y.unsqueeze(0), padding=(len(y) - 1) // 2)

    # equation 1 in the paper
    def _param_to_conc_patlak(self, x):
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 0.1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = torch.arange(0, t_end + step_size, step_size)

        t1_idx = torch.nonzero(t_samp)
        t1 = t_samp[t1_idx]
        dt = torch.diff(t1, dim=0)

        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)
        aifci_tens = torch.tensor(aifci, dtype=torch.float32)

        K_time = torch.cumsum(aifci_tens, dim=0) * dt[-1]

        mult = torch.stack((K_time, aifci_tens), 1)  # [1502, 2]
        mult_i = torch.zeros_like(mult)
        mult_c = mult + mult_i*1j

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))  # [2, 409600 (640 x 640)]
        yr = torch.matmul(mult_c, xr)  # [1502, 409600]

        oshape = [len(t_samp)] + self.ishape[1:]  # [1502, 1, 320, 320]
        yr = torch.reshape(yr, tuple(oshape))  # reshape yr to [1502, 1, 320, 320 ] --> concentration curve

        # Create a boolean mask for original sampling points
        logIdx = torch.zeros(len(t_samp), dtype=torch.bool)

        for i, t in enumerate(self.sample_time):
            closest_idx = (torch.abs(t_samp - t)).argmin().item()
            logIdx[closest_idx] = True

        conc = yr[logIdx, ...]  # back to shape [number of original sampling points, 320, 320]
        return conc

    def forward(self, x):
        x_c = torch.view_as_complex(x)
        self._check_ishape(x)

        # CA_dro = self._param_to_conc_tcm(x)
        #CA = self._param_to_conc_tcm_SnB(x)
        CA = self._param_to_conc_patlak(x_c)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        sig = CA_trans + (self.sig_baseline - self.M_steady)

        sig_maps = sig * self.csm
        ksp = self.ifft2c(sig_maps)
        return ksp


########################################################################################################################
TCM = False
Undersampled = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
DIR = os.path.dirname(os.path.realpath(__file__))
with h5py.File('Real Breast Data/GRASP-data/BC02_slice54g.mat', 'r') as file:
    # Extract the 'gdata' and 'cmaps' datasets
    kspace_data_raw = file['gdata'][:]
    coil_sensitivity_maps_raw = file['cmaps'][:]

kspace_data_complex = kspace_data_raw['real'] + 1j * kspace_data_raw['imag']
csm_data_complex = coil_sensitivity_maps_raw['real'] + 1j * coil_sensitivity_maps_raw['imag']
# rotate to match
rotated_coil_sensitivity_maps = np.rot90(csm_data_complex, k=1, axes=(1, 2))
rotated_csm_copy = rotated_coil_sensitivity_maps.copy() # eliminate negative strides

kspace_data = np.transpose(kspace_data_complex, (1, 0, 2, 3))

# Step 2: Downsample by taking the center 320x320 of the last two dimensions
center_crop = 320
start = (640 - center_crop) // 2  # Calculate start index for 320x320 crop

kspace_data_downsampled = kspace_data[:, :, start:start+center_crop, start:start+center_crop]

data_dro = sio.loadmat('DRO_data/Malignant/Run3_BC.mat')
breast_ksp_tens = torch.tensor(kspace_data_downsampled.copy(), dtype=torch.complex64, device=device)

baseline = sio.loadmat('Real Data/GRASP-data/BC_case1_single.mat')['cdata']
baseline_rearranged = np.transpose(baseline, (2, 0, 1))
baseline_downsampled = baseline_rearranged[:, start:start+center_crop, start:start+center_crop]
breast_ksp_baseline = torch.tensor(baseline_downsampled.copy(), dtype=torch.complex64, device=device)

# t1_map = data_dro['T10']
# t1_fixed = np.ones_like(t1_map)
# r1 = (1 / t1_map)
r1 = np.ones((320, 320))
aif = data_dro['aif']
concentration = data_dro['cts']
breast_img_tens = fft2c(breast_ksp_tens)
breast_img_baseline = fft2c(breast_ksp_baseline)
rotated_breast_img_baseline = torch.rot90(breast_img_baseline, k=1, dims=(1, 2)) # to match dims of breast_img_tens
meas_tens = breast_ksp_tens

# reading time array straight from DRO - t is in seconds !!
sample_time = sio.loadmat('Breast Data/t.mat')['t']
print('> sample time: ', sample_time)

# Baseline img - should have coil dim??
x0 = abs(rotated_breast_img_baseline[0, ...])
oshape = meas_tens.shape

if TCM:
    # TCM
    ishape = [4, 1] + list(oshape[-2:])

else:
    # Patlak
    ishape = [2, 1] + list(oshape[-2:]) + [2]  # extract the first two elements: [320, 320]

olen = np.prod(oshape)

x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device=device)

model = DCE(ishape, sample_time, x0, aif, r1, rotated_csm_copy, Undersampled, device=device)
#model = DCE(ishape, sample_time, x0, r1, aif, smap, Undersampled, device=device)

lossf = nn.MSELoss(reduction='sum')
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.001)
torch.autograd.set_detect_anomaly(True)

for epoch in range(100):
    # performs forward pass of the model with the current parameters 'x'
    fwd = model(x)
    check_for_nan(fwd, "Forward pass", epoch)

    # # print parameters at a specific pixel region - where it would be brightest for that parameter based on GT maps
    # ve_max = (x[0, 0, 270, 175])
    # vp_max = (x[1, 0, 210, 82])
    # fp_max = (x[2, 0, 270, 175])
    # PS_max = (x[3, 0, 280, 90])
    #
    # print(f've: {ve_max}, vp: {vp_max}, fp: {fp_max}, PS: {PS_max} ')

    # computes the MSE loss between the estimated kspace and the measured kspace data
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))
    #res = lossf(fwd, meas_tens)
    check_for_nan(res, "loss", epoch)

    # clears the gradients accumulated in the previous iteration
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)

    # computes the gradients of the loss wrt the model parameters
    res.backward()
    optimizer.step()  # parameter update
    # scheduler.step()

    print(f'Epoch {epoch}, Loss: {res.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

x_np = x.cpu().detach().numpy()

# fwd_img = fft2c(fwd)
fwd_np = fwd.cpu().detach().numpy()
print(x_np.dtype)
meas_np = meas_tens.detach().numpy()
outfilestr = 'BC_case_1_coils_run.h5'
f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('output', data=fwd_np)
f.close()
