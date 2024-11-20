import h5py
import os
import torch
import torch.fft as fft
from scipy.interpolate import pchip_interpolate
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
from scipy import signal
#from dce import DCE
import torch.nn.functional as F


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
                 R1,
                 Cp,
                 M0=5.0,
                 R1CA=4.30,
                 FA=10.,
                 TR= 4.87E-3,
                 x_iscomplex=True,
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
        # Cp_DCE = dce.arterial_input_function(sample_time)

        # read in breast sim aif, resize for shape [numpoints, ]
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)
        self.Cp = self.Cp.transpose(0, 1).view(-1)

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

    # equation 1 in the paper
    def _param_to_conc_patlak(self, x):
        # t1_idx = torch.nonzero(self.sample_time)
        # t1 = self.sample_time[t1_idx]

        # convert sample time to every 1 second
        t_end = self.sample_time[-1]
        step_size = 0.1
        t_step = torch.arange(0, t_end + step_size, step=step_size, dtype=torch.float32)

        dt = torch.diff(t_step, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]  # convolution - size: [22]

        mult = torch.stack((K_time, self.Cp), 1)  # [22, 2]

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))  # [2, 102400 (320*320)]
        yr = torch.matmul(mult, xr)  # [22, 102400]

        oshape = [len(self.sample_time)] + self.ishape[1:]  # [22, 1, 320, 320]
        yr = torch.reshape(yr, tuple(oshape))  # reshape yr to [22, 1, 320, 320 ] --> concentration curve
        return yr

    def fft_convolve_time_only(self, aif_map, kernel):
        # Initialize the result tensor with the appropriate shape
        result_shape = (aif_map.shape[0] + kernel.shape[0] - 1, aif_map.shape[1], aif_map.shape[2])
        conv_result = torch.zeros(result_shape, dtype=torch.float32, device=aif_map.device)

        # Perform 1D convolution along the time dimension for each spatial position
        for i in range(aif_map.shape[1]):
            for j in range(aif_map.shape[2]):
                # Extract the 1D signals
                aif_1d = aif_map[:, i, j].detach().numpy()
                kernel_1d = kernel[:, i, j].detach().numpy()

                # Perform the 1D convolution
                conv_1d = signal.fftconvolve(aif_1d, kernel_1d, mode='full')

                # Assign the result to the appropriate slice of the result tensor
                conv_result[:, i, j] = torch.tensor(conv_1d, dtype=torch.float32, device=aif_map.device, requires_grad=True)

        # The resulting tensor shape is [t_samp*2 - 1, 320, 320]
        return conv_result

    def _param_to_conc_tcm_SnB(self, parmap):
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 0.1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = torch.arange(0, t_end + step_size, step_size)
        aifci = torch.tensor(pchip_interpolate(self.sample_time.cpu().numpy(), self.Cp.cpu().numpy(), t_samp.cpu().numpy()), dtype=torch.float32, device=self.device)

        # Add delay to AIF
        delay_samples = int(3 / step_size)  # -- seconds delay
        aifci_delayed = torch.roll(aifci, delay_samples)
        aifci_delayed[:delay_samples] = 0  # zero out the initial values
        # aifci_tens = torch.tensor(aifci_delayed, dtype=torch.float32)

        #aifci_tens = torch.tensor(aifci, dtype=torch.float32)
        t_map = t_samp.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)
        aifMap = aifci_delayed.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)

        # x should have size [4, 1, x, y]
        ve = parmap[0, ...]
        vp = parmap[1, ...]
        fp = parmap[2, ...]
        ps = parmap[3, ...]

        Te = ve / ps
        T = (vp + ve) / fp
        Tc = vp / fp

        Te[torch.isnan(Te)] = 0
        T[torch.isnan(T)] = 0
        Tc[torch.isnan(Tc)] = 0

        theta_plus = ((T + Te) + torch.sqrt((T + Te) ** 2 - 4 * Tc * Te)) / (2 * Tc * Te)
        theta_minus = ((T + Te) - torch.sqrt((T + Te) ** 2 - 4 * Tc * Te)) / (2 * Tc * Te)

        theta_plus[torch.isnan(theta_plus)] = 0
        theta_minus[torch.isnan(theta_minus)] = 0

        he = theta_plus * theta_minus * (
                (torch.exp(-t_map * theta_minus) - torch.exp(
                    -t_map * theta_plus)) /
                (theta_plus - theta_minus))

        hp = theta_plus * theta_minus * (
                ((1 - Te * theta_minus) * torch.exp(
                    -t_map * theta_minus) +
                 (Te * theta_plus - 1) * torch.exp(
                            -t_map * theta_plus)) /
                (theta_plus - theta_minus))

        # Normalize he and hp so the integral (sum over time dimension) is 1
        he_sum = torch.sum(he, dim=0, keepdim=True)
        hp_sum = torch.sum(hp, dim=0, keepdim=True)

        he = he / he_sum
        hp = hp / hp_sum

        he[torch.isnan(he)] = 0
        hp[torch.isnan(hp)] = 0

        ce = self.fft_convolve_time_only(aifMap, he)
        cp = self.fft_convolve_time_only(aifMap, hp)

        # check shapes here
        conc = vp * cp[:len(t_samp), ...] + ve * ce[:len(t_samp), ...]

        # bool log indices where sampling occurred
        logIdx = np.zeros(len(t_samp))
        start_idx = 0
        for i in range(len(self.sample_time)):
            for j in range(start_idx, len(t_samp)):
                if self.sample_time[i] <= t_samp[j]:
                    logIdx[j] = 1
                    start_idx = j
                    break
        logIdx = logIdx.astype(bool)

        conc = conc[logIdx, ...]
        conc = conc.unsqueeze(1)
        return conc

    def forward(self, param):
        self._check_ishape(param)

        CA = self._param_to_conc_tcm_SnB(param)
        # CA = self._param_to_conc_patlak(param)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        sig = CA_trans + (self.sig_baseline - self.M_steady)
        ksp = self.ifft2c(sig)
        return sig

########################################################################################################################
TCM = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
DIR = os.path.dirname(os.path.realpath(__file__))
path = '../Breast Sim Pipeline'

data = sio.loadmat('DRO_data/Benign/Run22_PT.mat')

# TODO:
breast_img = data['IMG']
breast_img = np.transpose(breast_img, (2, 0, 1))
breast_img = breast_img[:, None, ...]
print('> img shape: ', breast_img.shape)

breast_img_tens = torch.tensor(breast_img.copy(), dtype=torch.float32, device=device)
t1_map = data['T10']
r1 = (1 / t1_map)
aif = data['aif']
concentration = data['cts']

breast_ksp_tens = ifft2c(breast_img_tens)
#meas_tens = breast_ksp_tens
meas_tens = breast_img_tens

# reading time array straight from DRO - t is in seconds !!
sample_time = sio.loadmat('Breast Data/t.mat')['t']
print('> sample time: ', sample_time)

# Baseline img
x0 = breast_img_tens[0, ...]
oshape = meas_tens.shape

if TCM:
    # TCM
    ishape = [4, 1] + list(oshape[-2:])

else:
    # Patlak
    ishape = [2, 1] + list(oshape[-2:])  # extract the first two elements: [320, 320]

olen = np.prod(oshape)

x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device=device)

# When using GT parmap from DRO
parmap_data = data['parMap']
parmap_r = parmap_data.transpose(2, 0, 1)
parmap_u = parmap_r[:, np.newaxis, ...]
parmap_tens = torch.tensor(parmap_u, dtype=torch.float32, requires_grad=True, device=device)

with torch.no_grad():
    parmap_tens[parmap_tens == 1e-8] = 0
    # x[0, ...] = 1E-4
    # x[1, ...] = 1E-4
    # x[2, ...] = 1E-4
    # x[3, ...] = 1E-4

# x = parmap_tens
model = DCE(ishape, sample_time, x0, r1, aif, device=device)
lossf = nn.MSELoss(reduction='sum')
optimizer = optim.Adam([x], lr=0.1, amsgrad=True)

for epoch in range(20):
    # performs forward pass of the model with the current parameters 'x'
    fwd = model(x)
    check_for_nan(fwd, "Forward pass", epoch)

    # print parameters at a specific pixel region - where it would be brightest for that parameter based on GT maps
    ve_max = (x[0, 0, 270, 175])
    vp_max = (x[1, 0, 210, 82])
    fp_max = (x[2, 0, 270, 175])
    PS_max = (x[3, 0, 280, 90])

    print(f've: {ve_max}, vp: {vp_max}, fp: {fp_max}, PS: {PS_max} ')

    # computes the MSE loss between the estimated kspace and the measured kspace data
    #res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))
    res = lossf(fwd, meas_tens)

    check_for_nan(res, "loss", epoch)

    # clears the gradients accumulated in the previous iteration
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)

    # computes the gradients of the loss wrt the model parameters
    res.backward()
    optimizer.step()  # parameter update
    print('> epoch %3d loss %.9f' % (epoch, res.item() / olen))

x_np = x.cpu().detach().numpy()
print(x_np.dtype)

meas_np = meas_tens.detach().numpy()

outfilestr = 'SNB_test_ksp2.h5'

f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('meas', data=meas_np)

f.close()
