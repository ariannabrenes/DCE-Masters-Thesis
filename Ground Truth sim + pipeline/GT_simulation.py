import h5py
import os
import torch

import numpy as np
import torch.nn as nn
import torch.fft as fft
import dce
import scipy.io as sio
from scipy.io import savemat

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
                 R1 = 1.,
                 M0 = 5.,
                 R1CA = 4.39,
                 FA = 15.,
                 TR = 0.006,
                 mask = None):
        super(DCE, self).__init__()

        self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32)
        self.sig_baseline = torch.tensor(sig_baseline, dtype=torch.float32)

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32)

        self.FA_radian = self.FA * np.pi / 180.

        E1 = torch.exp(-self.TR * self.R1)

        topM0 = self.sig_baseline * (1 - torch.cos(self.FA_radian) * E1)
        bottomM0 = torch.sin(self.FA_radian) * (1 - E1)
        self.M0 = topM0 / bottomM0

        self.M0_trans = self.M0 * torch.sin(self.FA_radian)

        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))
        #self.csm = csm


        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32)

        self.mask = torch.tensor(mask, dtype = torch.float32, device = device)

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))


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

    def _param_to_conc(self, x):
        t1_idx = torch.nonzero(self.sample_time)
        t1 = self.sample_time[t1_idx]
        dt = torch.diff(t1, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]

        mult = torch.stack((K_time, self.Cp), 1)

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))

        yr = torch.matmul(mult, xr)

        oshape = [len(self.sample_time)] + self.ishape[1:]
        yr = torch.reshape(yr, tuple(oshape))

        return yr

    def forward(self, x):
        if torch.is_tensor(x) is not True:
            x = torch.tensor(x, dtype=torch.float32)

        self._check_ishape(x)

        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x)

        # concentration to MR signal
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))

        # transient concentration?
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        # try setting the signal baseline to same as M_steady
        self.sig_baseline = self.M_steady

        # y is the signal here
        sig = CA_trans + self.sig_baseline - self.M_steady

        # multiply coil with signal
        #sig_csm = sig * self.csm

        # return in kspace
        y = self.ifft2c(sig)

        #y = self.mask * y

        return y

    def get_conc(self, x):
        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x)
        return CA

    def get_sig(self, x):
        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        # y is the signal
        sig = CA_trans + self.sig_baseline - self.M_steady

        return sig

# %%
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

path = '../Ground Truth sim + pipeline'

N_time = 50
N_t0 = 8

# units are in seconds!
delay = np.zeros(N_t0)
injection = np.arange(1, N_time - N_t0 + 1, 1) * 5/60
time = np.concatenate((delay, injection))
#sample_time = time - delay
sample_time = time[N_t0:]

# csm = sio.loadmat('GT_cmaps_sq.mat')['cmaps']
# csm = torch.tensor(csm, dtype=torch.complex64, device=device)

# read in Umask
Umask = sio.loadmat('../USC data + pipeline/Umask.mat')['U1']
Umask = np.transpose(Umask, axes = (3, 2, 4, 1, 0))
Umask = np.fft.ifftshift(Umask, axes= (-2, -1))

oshape = [N_time - N_t0, 1, 1, 256, 256]
ishape = [2] + list(oshape[1:])

img_len = np.prod(oshape[1:])
k_trans = (np.arange(img_len).reshape(ishape[1:])) / img_len * 0.5

# have v_p gradient going from top to bottom
ishape_vp = [2, 1, 1, 256, 256]
v_p = (np.arange(img_len).reshape(ishape_vp[1:])) / img_len * 0.5
v_p = np.swapaxes(v_p, -2, -1)

gt = torch.zeros(ishape, dtype=torch.float32)
gt[0, ...] = torch.tensor(k_trans)
gt[1, ...] = torch.tensor(v_p)

UNDERSAMPLED = False
Umask = Umask

if UNDERSAMPLED:
   mask = Umask

else:
    mask = np.ones_like(Umask)

# signal baseline? set in forward pass?
sig_baseline = torch.ones([1, 256, 256])
#model = DCE(ishape, sample_time, sig_baseline, mask = mask, csm=csm).to(device)

# passing whole time through:
model = DCE(ishape, time, sig_baseline, mask = mask).to(device)

# sim in image space
sim = model(gt)
img = fft2c(sim)
conc = model.get_conc(gt)

# add noise to image space - w/ baseline points?
noise = torch.randn([50, 1, 1, 256, 256]) * 0.01

# add noise to kspace
split = torch.view_as_real(sim)

# add noise to both real and imag parts of kspace
real = split[..., 0] + noise
imag = split[..., 1] + noise

# put back together
sim_noise = real + 1j * imag
img_noise = fft2c(sim_noise)

sim_np = sim_noise.cpu().detach().numpy()
gt_np = gt.cpu().detach().numpy()

if UNDERSAMPLED:
    savemat(path + "/gt_ksp_undersamp_noise_0.01.mat", {'GTksp':sim_np})
else:
    savemat(path + "/gt_ksp_newM0_sig1.mat", {'GTksp': sim_np})

# gt_img = model.get_sig(gt)
# savemat("test_gt_img_undersamp.mat", {'GTimg':gt_img.cpu().detach().numpy()})

outfilestr = '/GT_TKmaps_newM0.h5'

f = h5py.File(path + outfilestr, 'w')
f.create_dataset('param', data=gt_np)
f.create_dataset('gt', data=sim_np)
f.close()
