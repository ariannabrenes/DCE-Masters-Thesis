import h5py
import os
import torch
import torch.fft as fft

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim

from dce import DCE
import dce


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
                 x_iscomplex=True):
        super(DCE, self).__init__()

        if x_iscomplex:
            self.ishape = list(ishape[:-1])
        else:
            self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32)
        self.sig_baseline = torch.tensor(sig_baseline, dtype=torch.float32)

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32)

        self.FA_radian = self.FA * np.pi / 180.
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)

        E1 = torch.exp(-self.TR * self.R1)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32)

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def center_kspace(self, k, axes=(-2, -1)):
        c = torch.fft.fftshift(k, dim=axes)
        return c

    def matmul_complex(self, torch1, t2):
        return torch.view_as_complex(
            torch.stack((torch1.real @ t2.real - torch1.imag @ t2.imag, torch1.real @ t2.imag + torch1.imag @ t2.real),
                        dim=2))

    # equation 1 in the paper
    def _param_to_conc(self, x):
        t1_idx = torch.nonzero(self.sample_time)
        t1 = self.sample_time[t1_idx]
        dt = torch.diff(t1, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]  # convolution

        mult = torch.stack((K_time, self.Cp), 1)
        # mult = torch.view_as_complex(mult)

        mult_i = torch.zeros_like(mult)
        mult_c = mult + 1j * mult_i

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))

        # yr = torch.matmul(mult, xr) # matmul does not take complex
        yr = self.matmul_complex(mult_c, xr)

        oshape = [len(self.sample_time)] + self.ishape[1:]
        yr = torch.reshape(yr, tuple(oshape))
        return yr

    # shape of x should be [2, 150, 256]
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
        # smaps shape [42, 8, 150, 256]
        #change y to image space here?
            # now smaps and y are in image space
        # TODO: Problem here is that y_ifft is not in image space
        y_ifft = ifft2c(y)

        # MULTIPLY COILS IN IMAGE SPACE - the issue here is that sMaps_DCE is in kspace
        sMaps_DCE = sMaps_squeezed * y_ifft
        #sMaps_DCE = sMaps_squeezed * y

        # uMask * FFT(sMaps * DCE) --> [42, Nc, Nx, Ny]
        # change back to kspace
        fft_smaps = fft2c(sMaps_DCE)

        # y shape: [42, 1, Nc, Nx, Ny]
        # apply the undersampling mask
        #y = identity_mask * fft_smaps
        y = shift_Umask * fft_smaps
        return y


########################################################################################################################
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

DIR = os.path.dirname(os.path.realpath(__file__))

def fft2c(k, axes=(-2, -1)):
    k = fft.ifftshift(k, dim=axes)
    k = fft.fft2(k, dim=axes)
    k = fft.fftshift(k, dim=axes)
    scale = 1 / torch.sqrt(torch.prod(torch.tensor(k.shape[-2:])))
    return scale * k

def ifft2c(x, axes=(-2, -1)):
    x = fft.ifftshift(x, dim=axes)
    x = fft.ifft2(x, dim=axes)
    x = fft.fftshift(x, dim=axes)
    scale = torch.sqrt(torch.prod(torch.tensor(x.shape[-2:])))
    return scale * x

# TODO:
# kspace from MATLAB already multiplied by the sMaps?
# 1st change the fimg to fully sampled k space data
# 2nd change the fimg to the undersampled kspace data

# Remember: cannot multiply coils in kspace, only in image space

#f = sio.loadmat('k.mat')['k']  # fully sampled
f = sio.loadmat('kU.mat')['kU'] # undersampled
f_squeeze = np.squeeze(f)
f1 = np.transpose(f_squeeze, axes=(2,3,1,0))
f = torch.tensor(f1, dtype=torch.complex64)

# TODO: Notes
    # fimg at this point is in kspace for direct recon in kspace
f_center = fft2c(torch.fft.ifft2(f, norm='ortho'))
fimg = f_center

# read in data Umask and sMaps
Umask = sio.loadmat('USC data + pipeline/Umask.mat')['U1']
Umask = np.transpose(Umask, axes = (3, 2, 4, 1, 0))
Umask = np.fft.ifftshift(Umask, axes= (-2, -1))

# complex type
sMaps = sio.loadmat('USC data + pipeline/sMaps.mat')['sMaps']
sMaps = torch.tensor(sMaps, dtype=torch.complex64)

# Transpose the dimensions to [1, 1, Nc, Nx, Ny] - for torch as real IN FWD
transposed_sMaps = sMaps.permute(*((2, 3, 4, 1, 0)))

# if extra 1 is not needed: squeeze the first dimension. Shape = [1, Nc, Nx, Ny]
sMaps_squeezed = transposed_sMaps.squeeze(dim=1)

N_time = fimg.shape[0]

N_t0 = 8
tim1 = np.zeros(N_t0)
tim2 = np.arange(1, N_time - N_t0 + 1, 1) * 5 / 60

sample_time = np.concatenate((tim1, tim2))
sample_time = sample_time[N_t0:]

print('> sample time: ', sample_time)

# this is the measured experiment not including the baseline measurements (first 8)
meas_exp = fimg[N_t0:]

print('> fimg shape ', fimg.shape)

EXP = True
# if false, this is the ground truth

if EXP:
    meas = torch.tensor(meas_exp, dtype=torch.complex64)
    meas = meas[:, None, ...]

# TODO: Notes
    # x0 needs to be in image space, the baseline images without coils
    # debug: x0 is still a multi coil image here, need to do a coil combination on x0

    #Coil combination?
    fimg_abs_squared = torch.abs(fimg)**2
    fimg_sum = torch.sum(fimg_abs_squared, dim=1, keepdim=True)
    fimg_combination = torch.sqrt(fimg_sum)

    x0 = ifft2c(fimg_combination[0, ...])

    oshape = meas.shape
    ishape = [2, 1, 1] + list(oshape[-2:]) + [2]
    # ishape = [2] + list(oshape[1:-1])

else:
    # plot from 0 to 10 (vmin and vmax )
    oshape = [N_time, 1, 1, 64, 64]
    ishape = [2] + list(oshape[1:])

    img_len = np.prod(oshape[1:])

    # change ktrans values here, suggestions from Maher?
    k_trans = (np.arange(img_len).reshape(ishape[1:])) / img_len * 10
    v_p = (np.arange(img_len).reshape(ishape[1:])) / img_len * 0.02

    gt = torch.zeros(ishape, dtype=torch.float32)
    gt[0, ...] = torch.tensor(k_trans)
    gt[1, ...] = torch.tensor(v_p)

    model = DCE(ishape, sample_time, 0).to(device)

    # reshape the ground truth tensor to match shapes
    ground_truth_model = model(gt)
    tens_rand = torch.rand(oshape)  # random noise
    reshape_tens = tens_rand[:42, ...]

    # measured signal from simulation
    meas = ground_truth_model + reshape_tens * 0.001
    x0 = 0

olen = np.prod(oshape)
# cast as a complex here ?
x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True)

# x = torch.zeros(ishape, dtype=torch.complex64,
#                 requires_grad=True)

model = DCE(ishape, sample_time, x0).to(device)
lossf = nn.MSELoss(reduction='sum')
# optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)

for epoch in range(10):
    fwd = model(x)
    # call view as real here on both
    # another option - define own loss function
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas))

    optimizer.zero_grad()
    res.backward()
    optimizer.step()

    print('> epoch %3d loss %.6f' % (epoch, res.item() / olen))

x_np = x.cpu().detach().numpy()
print(x_np.dtype)

if EXP:
    outfilestr = 'torch_sgd_exp.h5'
else:
    outfilestr = 'torch_sgd_sim.h5'

f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('meas', data=fimg)

if EXP is False:
    f.create_dataset('gt', data=gt.cpu().detach().numpy())

f.close()
