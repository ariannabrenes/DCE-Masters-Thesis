import h5py
import os
import torch
import torch.fft as fft

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
# from sigpy import linop
# import sigpy as sp

from dce import DCE
import dce
from timeit import default_timer as timer


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
                 R1=1.,
                 M0=5.,
                 R1CA=4.39,
                 FA=15.,
                 TR=0.006,
                 x_iscomplex = True,
                 csm = None,
                 Umask = None,
                 undersampled = None,
                 device = torch.device('cuda')):
        super(DCE, self).__init__()

        if x_iscomplex:
            self.ishape = list(ishape[:-1])
        else:
            self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32, device=device)
        self.sig_baseline = torch.tensor(sig_baseline, dtype=torch.float32, device=device)

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32, device=device)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32, device=device)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32, device=device)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32, device=device)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32, device=device)

        self.FA_radian = self.FA * np.pi / 180.
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)

        E1 = torch.exp(-self.TR * self.R1)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)

        self.csm = torch.tensor(csm, dtype=torch.complex64, device=device)
        self.Umask = torch.tensor(Umask, dtype=torch.float32, device=device)

        self.undersampled = undersampled 

        self.device = device


    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def matmul_complex(self, torch1, t2):
        return torch.view_as_complex(
            torch.stack((torch1.real @ t2.real - torch1.imag @ t2.imag, torch1.real @ t2.imag + torch1.imag @ t2.real), dim=2))

    def fft2c(self, x, axes=(-2, -1), norm='ortho'):
        x = torch.fft.ifftshift(x, dim=axes)
        x = fft.fft2(x, dim=axes, norm=norm)

        # center the kspace
        x = torch.fft.fftshift(x, dim=axes)
        return x

    # equation 1 in the paper
    def _param_to_conc(self, x):
        t1_idx = torch.nonzero(self.sample_time)
        t1 = self.sample_time[t1_idx]
        dt = torch.diff(t1, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]  # convolution

        mult = torch.stack((K_time, self.Cp), 1)
        mult_i = torch.zeros_like(mult)
        mult_c = mult + mult_i*1j

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))
        yr = self.matmul_complex(mult_c, xr)

        oshape = [len(self.sample_time)] + self.ishape[1:]
        yr = torch.reshape(yr, tuple(oshape))
        return yr

    # shape of x should be [2, y, x]
    def forward(self, x):
        x_c = torch.view_as_complex(x)
        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x_c)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        # y is the signal
        y = CA_trans + self.sig_baseline - self.M_steady

        # TODO: Implement SENSE
        sMaps_DCE = self.csm * y
        fft_smaps = self.fft2c(sMaps_DCE)

        if self.undersampled:
            y = self.Umask * fft_smaps

        else:
            identity_mask = torch.ones_like(self.Umask)
            y = identity_mask*fft_smaps

        return y

########################################################################################################################
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
DIR = os.path.dirname(os.path.realpath(__file__))
dir_path = '/content/drive/MyDrive/Colab Notebooks'

# TODO:
# Read in the DCE images
f = h5py.File('DCE.h5', 'r')
dce_original = f['img'][:]
dce_original = dce_original[:, None, ...]  # 5 dimensions
dce_original_tens = torch.tensor(dce_original, dtype=torch.float64, device=device)
f.close()

print('> dce shape: ', dce_original.shape)

# Load in data outside of the forward pass
Umask = sio.loadmat('Umask.mat')['U1']
Umask = np.transpose(Umask, axes=(3, 2, 4, 1, 0))
Umask = np.fft.ifftshift(Umask, axes=(-2, -1))

print('> Umask shape: ', Umask.shape)

csm = sio.loadmat('sMaps.mat')['sMaps']
csm = np.transpose(np.squeeze(csm), axes=(2, 1, 0))
csm_tens = torch.tensor(csm, dtype=torch.complex64, device=device)
print('> csm shape: ', csm.shape)

# multiply image with sMaps to get the coil maps
csm_dce = dce_original_tens * csm_tens

# fftc to get kspace data - make sure its centered using fftc
ksp = ifft2c(csm_dce, axes=(-2, -1))

N_time = ksp.shape[0]
N_t0 = 8
delay = np.zeros(N_t0)
injection = np.arange(1, N_time - N_t0 + 1, 1) * 5 / 60

sample_time = np.concatenate((delay, injection))
sample_time = sample_time[N_t0:]

print('> sample time: ', sample_time)

# Set to true to undersample the kspace 
UNDERSAMPLED = False

if UNDERSAMPLED:
  ksp_usamp = Umask * ksp
  meas_exp = ksp_usamp[N_t0:]
  Umask = Umask[N_t0:]

else:
  meas_exp = ksp[N_t0:]
  Umask = Umask[N_t0:]

print('> Kspace shape ', ksp.shape)

EXP = True
# if false, this is the ground truth

if EXP:
    meas_tens = torch.tensor(meas_exp,dtype=torch.complex64, device=device)
    x0 = dce_original[0, ...]

    oshape = meas_exp.shape
    ishape = [2, 1, 1] + list(oshape[-2:]) + [2]
    # ishape = [2] + list(oshape[1:-1])

else:
    # plot from 0 to 10 (vmin and vmax)
    oshape = [N_time, 1, 1, 64, 64]
    #ishape = [2] + list(oshape[1:])
    ishape = [2, 1, 1] + list(oshape[-2:]) + [2]

    img_len = np.prod(oshape[1:])

    # k trans value was too high with 10
    k_trans = (np.arange(img_len).reshape(ishape[1:])) / img_len * 1
    v_p = (np.arange(img_len).reshape(ishape[1:])) / img_len * 0.02

    gt = torch.zeros(ishape, dtype=torch.float32)
    gt[0, ...] = torch.tensor(k_trans)
    gt[1, ...] = torch.tensor(v_p)

    model = DCE(ishape, sample_time, 0, csm = csm, Umask = Umask, device = device)

    # reshape the ground truth tensor to match shapes
    ground_truth_model = model(gt)
    tens_rand = torch.rand(oshape)  # random noise

    # measured signal from simulation
    meas_tens = ground_truth_model + tens_rand * 0.001
    x0 = 0


olen = np.prod(oshape)
x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device = device)

model = DCE(ishape, sample_time, x0, csm = csm, Umask = Umask, undersampled = UNDERSAMPLED, device = device)
lossf = nn.MSELoss(reduction='sum')
#optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)

start = timer()
for epoch in range(10):
    # performs forward pass of the model with the current parameters 'x'
        # generating estimated kspace data
    fwd = model(x)

    # computes the MSE loss between the estimated kspace and the measured
        #kspace data
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))

    # clears the gradients accumulated in the previous iteration
    optimizer.zero_grad()
    #res.requires_grad = True

    # computes the gradients of the loss wrt the model parameters
    res.backward()
    optimizer.step() # parameter update

    print('> epoch %3d loss %.9f' % (epoch, res.item() / olen))

end = timer()
print(end - start)

x_np = x.cpu().detach().numpy()
print(x_np.dtype)

#savemat("gt_ksp_undersamp_noise_0.01.mat", {'GTksp': x_np})

if EXP:
    outfilestr = 'torch_sgd_exp.h5'
else:
    outfilestr = 'torch_sgd_sim.h5'

f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('meas', data=meas_exp)

if EXP is False:
    f.create_dataset('gt', data=gt.cpu().detach().numpy())

f.close()
