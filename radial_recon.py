import h5py
import os
import torch
import torch.fft as fft

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import sigpy as sp

from dce import DCE
import dce
import torchkbnufft as tkbn


def fftc(input, axes=None, norm='ortho'):
    tmp = np.fft.ifftshift(input, axes=axes)
    tmp = np.fft.fftn(tmp, axes=axes, norm=norm)
    output = np.fft.fftshift(tmp, axes=axes)
    return output


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
                 Umask=None,
                 traj = None,
                 device=torch.device('cpu')):
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
        self.TR = torch.tensor(np.array(TR), dtype=torch.float64, device=device)

        self.FA_radian = self.FA * np.pi / 180.
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)

        E1 = torch.exp(-self.TR * self.R1)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)

        self.csm = torch.tensor(csm, dtype=torch.complex64, device=device)
        self.Umask = torch.tensor(Umask, dtype=torch.float32, device=device)

        self.traj = torch.tensor(traj, dtype=torch.float64, device=device)

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
        mult_c = mult + 1j * mult_i

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))
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
        sMaps_DCE = self.csm * y

        if traj.any():
            y_size = y[0, 0, ...]
            y_size = y_size.type(torch.float32)
            im_size = y_size.detach().numpy()

            dcf = tkbn.calc_density_compensation_function(ktraj=self.traj, im_size=im_size)
            #dcf = (self.traj[..., 0]**2 + self.traj[..., 1]**2)**0.5
            #dcf_reshape = dcf.reshape(1, -1).repeat(2,1).transpose(1, 0)

            # create NUFFT objects
            nufft_ob = tkbn.KbNufft(im_size = im_size).to(torch.device('cpu'))
            adjnufft_ob = tkbn.KbNufftAdjoint(im_size= im_size).to(torch.device('cpu'))

            # calculate k-space data
            kdata = nufft_ob(y, self.traj)
            image = adjnufft_ob(kdata * dcf, self.traj)

        else:
            fft_smaps = self.fft2c(sMaps_DCE)

            if UNDERSAMPLED:
                y = self.Umask * fft_smaps
            else:
                identity_mask = torch.ones_like(self.Umask)
                y = identity_mask * fft_smaps

        return y


########################################################################################################################
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

DIR = os.path.dirname(os.path.realpath(__file__))
#dir_path = '/content/drive/MyDrive/Colab Notebooks'

# Read in the DCE images
# f = h5py.File('radialx_ksp_n.mat', 'r')
# dce_original = f['img'][:]
dce_radial = sio.loadmat('radialx_ksp_n.mat')['ksp']
print('> dce shape: ', dce_radial.shape)

# dce_radial already has coil information

# set Ntime and sample time
N_time = dce_radial.shape[0]
N_t0 = 8
tim1 = np.zeros(N_t0)
tim2 = np.arange(1, N_time - N_t0 + 1, 1) * 5 / 60

sample_time = np.concatenate((tim1, tim2))
sample_time = sample_time[N_t0:]
print('> sample time: ', sample_time)

# read in sensitivity maps
Umask = sio.loadmat('USC data + pipeline/Umask.mat')['U1']
Umask = np.transpose(Umask, axes=(3, 2, 4, 1, 0))
Umask = np.fft.ifftshift(Umask, axes=(-2, -1))

print('> Umask shape: ', Umask.shape)

csm = sio.loadmat('USC data + pipeline/sMaps.mat')['sMaps']
csm = np.transpose(np.squeeze(csm), axes=(2, 1, 0))
print('> csm shape: ', csm.shape)

# True --> Use radial data, False --> Cartesian data
RADIAL = True

if RADIAL:
    #f.close()
    # Obtain trajectory 13 spokes
    N_coil = 8
    N_x = 256
    N_spokes = 13
    N_tot_spokes = N_spokes * N_time
    base_res = N_x
    N_samples = base_res * 2
    base_lin = np.arange(N_samples).reshape(1, -1) - base_res
    tau = 0.5 * (1 + 5 ** 0.5)
    base_rad = np.pi / (2 + tau - 1)
    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    # resize the dce and the csm
    csm_resize = sp.resize(csm, [8, 256, 256])
    #dce_original = sp.resize(dce_original, [50, 1, 256, 256])
    #dce_original_csm = dce_original * csm_resize
    orig_ksp = sp.fft(dce_radial, axes=(-2, -1))
    meas_exp = orig_ksp[N_t0:]

    # obtain the trajectory
    traj_t = traj.reshape(N_time, N_spokes, N_samples, 2)
    traj_r = traj_t[10, ...]


else:
    dce_original = dce_original[:, None, ...]  # 5 dimensions
    f.close()

    # multiply image with sMaps to get the coils
    csm_dce = dce_original * csm

    # fftc to get kspace data - make sure its centered using fftc
    ksp = fftc(csm_dce, axes=(-2, -1))

    # set to true to undersample the kspace with Umask
    UNDERSAMPLED = True
    Umask = Umask[N_t0:]

    if UNDERSAMPLED:
        ksp_usamp = Umask * ksp[N_t0:]
        meas_exp = ksp_usamp

    else:
        meas_exp = ksp[N_t0:]

    print('> Kspace shape ', ksp.shape)


EXP = True
# if false, this is the ground truth

if EXP:
    meas_tens = torch.tensor(meas_exp, dtype=torch.complex64, device=device)
    x0 = dce_original[0, ...]

    oshape = meas_exp.shape
    ishape = [2, 1] + list(oshape[-2:]) + [2]
    # ishape = [2] + list(oshape[1:-1])

else:
    # plot from 0 to 10 (vmin and vmax)
    oshape = [N_time, 1, 1, 64, 64]
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

# TODO:
# use mse metrics to compare, skimage mse
# simulate real radial undersampled data based on the DCE images
# change the fft to non cartesian --> nufft

olen = np.prod(oshape)
x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device=device)

#model = DCE(ishape, sample_time, x0, csm = csm, Umask = Umask,  device = device)
model = DCE(ishape, sample_time, x0, csm=csm_resize, Umask=Umask, traj=traj_r, device=device)

lossf = nn.MSELoss(reduction='sum')
# optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)

for epoch in range(200):
    fwd = model(x)
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))

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
f.create_dataset('meas', data=meas_exp)

if EXP is False:
    f.create_dataset('gt', data=gt.cpu().detach().numpy())

f.close()
