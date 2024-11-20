import h5py
import os
import torch
import torch.fft as fft
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
from numpy.fft import fftshift
import dce

#
# def arterial_input_function(sample_time,
#                             A=[0.309, 0.330],
#                             T=[0.17046, 0.365],
#                             sigma=[0.0563, 0.132],
#                             alpha=1.050,
#                             beta=0.1685,
#                             s=38.078,
#                             tau=0.483,
#                             Hct=0.4):
#     """
#     Args:
#         sample_time (array): sampling time array for AIF calculation. [unit: minutes]
#
#         Please refer to the following references for the definition of other parameters.
#
#     References:
#         * Parker GJM, Roberts C, Macdonald A, Buonaccorsi GA, Cheung S, Buckley DL, Jackson A, Watson Y, Davies K, Jayson GC.
#           Experimentally-derived functional form for a population-averaged high-temporal-resolution arterial input function for dynamic contrast-enhanced MRI.
#           Magnetic Resonance in Medicine 56:993-1000 (2006).
#
#         * Tofts PS, Berkowitz B, Schnall MD.
#           Quantitative analysis of dynamic Gd-DTPA enhancement in breast tumors using a permeability model.
#           Magnetic Resonance in Medicine 33:564-568 (1995).
#
#         * https://mriquestions.com/uploads/3/4/5/7/34572113/dce-mri_siemens.pdf
#     """
#
#     sample_mask = sample_time > 0
#
#     Cp = np.zeros_like(sample_time)
#
#     # sigmoid function
#     exp_b = np.exp(-beta * sample_time)
#     exp_s = np.exp(-s * (sample_time - tau))
#     sigmoid_vals = alpha * exp_b / (1 + exp_s)
#
#     # Gaussian functions
#     for n in range(len(A)):
#         scale = A[n] / (sigma[n] * (2 * np.pi) ** 0.5)
#         exp_t = np.exp(-((sample_time - T[n]) ** 2.) / (2 * sigma[n] ** 2))
#         Cp += scale * exp_t
#
#     Cp += sigmoid_vals
#
#     Cp *= sample_mask
#
#     Cp /= 3  # scaling
#
#     Cp /= (1 - Hct)
#
#     return Cp

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
                 x_iscomplex=True,
                 csm=None,
                 mask =None,
                 device = torch.device('cpu')):

        super(DCE, self).__init__()

        if x_iscomplex:
            self.ishape = list(ishape[:-1])
        else:
            self.ishape = list(ishape)

        self.device = device
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

        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)
        self.csm = csm
        self.mask = torch.tensor(mask, dtype=torch.float32, device=device)

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def matmul_complex(self, torch1, t2):
        return torch.view_as_complex(
            torch.stack((torch1.real @ t2.real - torch1.imag @ t2.imag, torch1.real @ t2.imag + torch1.imag @ t2.real),
                        dim=2))

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
    def _param_to_conc(self, x):
        t1_idx = torch.nonzero(self.sample_time)
        t1 = self.sample_time[t1_idx]
        dt = torch.diff(t1, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]  # convolution
        mult = torch.stack((K_time, self.Cp), 1)
        mult_i = torch.zeros_like(mult)
        mult_c = mult + 1j * mult_i

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))
        yr = self.matmul_complex(mult_c, xr)
        oshape = [len(self.sample_time)] + self.ishape[1:]
        yr = torch.reshape(yr, tuple(oshape))

        return yr

        # shape of x should be [2, 150, 256]
    def forward(self, x):
        x_c = torch.view_as_complex(x)

        self._check_ishape(x)

        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x_c)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        #self.sig_baseline = self.M_steady

        # y is the signal
        sig = CA_trans + self.sig_baseline - self.M_steady

        # TODO: Implement SENSE
        sig_csm = self.csm * sig
        #fft_smaps = self.fft2c(sMaps_DCE)
        #y = fft_smaps * self.mask
        #y = fft_smaps
        y = self.ifft2c(sig_csm)
        return y

    def get_conc(self, x):
        x_c = torch.view_as_complex(x)
        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x_c)
        return CA

    def get_sig(self, x):
        x_c = torch.view_as_complex(x)

        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x_c)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        # y is the signal
        sig = CA_trans + self.sig_baseline - self.M_steady

        return sig

########################################################################################################################

if torch.cuda.is_available():

    device = "cuda"

else:

    device = "cpu"

print(device)
path = '../Ground Truth sim + pipeline'

# TODO:
Umask = sio.loadmat('../USC data + pipeline/Umask.mat')['U1']
Umask = np.transpose(Umask, axes=(3, 2, 4, 1, 0))
Umask = np.fft.ifftshift(Umask, axes=(-2, -1))
Umask = Umask[:, :, :4, ...]

print('> Umask shape: ', Umask.shape)

csm = sio.loadmat('GT_cmaps_sq.mat')['cmaps']
csm = torch.tensor(csm, dtype=torch.complex64, device=device)
#csm = np.ones_like(csm)
#csm = np.ones((4, 256, 256))
#csm = np.transpose(np.squeeze(csm), axes=(2, 1, 0))
print('> csm shape: ', csm.shape)

N_time = 50
N_t0 = 8
tim1 = np.zeros(N_t0)
tim2 = np.arange(1, N_time - N_t0 + 1, 1) * 5 / 60
time = np.concatenate((tim1, tim2))
sample_time = time[N_t0:]

print('> sample time: ', sample_time)

# Set to true to undersample the kspace

UNDERSAMPLED = False
Umask = Umask[N_t0:]

if UNDERSAMPLED:
    # ksp_usamp = Umask * ksp
    # meas_exp = ksp_usamp[N_t0:]
    mask = Umask
    gt_ksp_usamp_load = sio.loadmat('test_gt_ksp_undersamp.mat')['GTksp']
    gt_img = fft2c(gt_ksp_usamp_load)
    gt_img_csm = gt_img * csm
    meas_exp = fft2c(gt_img_csm)

else:
    # gt_img_load = sio.loadmat('test_gt_img_0.mat')['GTksp']
    # gt_img = gt_img_load * csm_ones
    # gt_ksp = fftc(gt_img)

    gt_ksp_load = sio.loadmat('gt_ksp_newM0_sig1.mat')['GTksp']
    #multiply csm in img space!!!

    gt_ksp_load = torch.tensor(gt_ksp_load, dtype=torch.complex64, device=device)
    gt_img = fft2c(gt_ksp_load)
    gt_img_csm = gt_img * csm

    gt_ksp = ifft2c(gt_img_csm)
    meas_exp = gt_ksp
    meas_exp = meas_exp[N_t0:]
    mask = np.ones_like(Umask)

print('> Kspace shape ', meas_exp.shape)

EXP = True
# If false, this is the ground truth

if EXP:
    meas_tens = meas_exp
    #x0 = gt_img[0, ...] # set signal baseline here - first frame in image space
    x0 = gt_img[0, ...]

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

    model = DCE(ishape, sample_time, 0, csm=csm, Umask=Umask, device=device)

    # reshape the ground truth tensor to match shapes

    ground_truth_model = model(gt)
    tens_rand = torch.rand(oshape)  # random noise
    reshape_tens = tens_rand[:42, ...]

    # measured signal from simulation
    meas = ground_truth_model + reshape_tens * 0.001
    x0 = 0

olen = np.prod(oshape)

x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device=device)

model = DCE(ishape, sample_time, x0, csm=csm, mask = mask, device=device)
lossf = nn.MSELoss(reduction='sum')

# optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)

for epoch in range(300):
    fwd = model(x)
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))
    optimizer.zero_grad()
    # res.requires_grad = True
    res.backward()
    optimizer.step()

    print('> epoch %3d loss %.9f' % (epoch, res.item() / olen))

sig = model.get_sig(x)
conc = model.get_conc(x)

x_np = x.cpu().detach().numpy()

print(x_np.dtype)

if EXP:
    outfilestr = '/TK_maps_noise_csm4_newM0_sig1.h5'

else:

    outfilestr = 'torch_sgd_sim.h5'

f = h5py.File(path + outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('meas', data=meas_exp)

if EXP is False:
    f.create_dataset('gt', data=gt.cpu().detach().numpy())

f.close()




