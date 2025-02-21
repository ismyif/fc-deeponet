import numpy as np
import skfmm
from tqdm import *
# import segyio
import time

data = np.load('velmodels_valid_read.npz')
velmodels = data['velmodels'] / 1000
print("velmodels.shape:",velmodels.shape)

zmin = 0.;
zmax = 1.38;
deltaz = 0.02;
xmin = 0.;
xmax = 1.38;
deltax = 0.02;

z = np.arange(zmin, zmax + deltaz, deltaz)
nz = z.size

x = np.arange(xmin, xmax + deltax, deltax)
nx = x.size

Z, X = np.meshgrid(z, x, indexing='ij')
Zra, Xra, Zsa, Xsa, Vsa, Tref, tauref, = [], [], [], [], [], [], []

input_data = np.empty((0,1,70,70))
tau_data = np.empty((0,1,70,70))
XXs = np.empty((0,2))
for i in tqdm(range(0, len(velmodels))):
    vel = velmodels[i, :, :]

    selected_pts1 = np.random.choice(a=len(Z) - 1, size=4,replace=False)
    selected_pts2 = np.random.choice(a=len(Z) - 1, size=4, replace=False)
    Zr = Z
    Xr = X

    Zs = np.concatenate((np.array([0]), np.array([0]),np.array([0]), np.array([0])),0)
    Xs = X[0][selected_pts1]

    for ns, (szi, sxi) in enumerate(zip(Zs, Xs)):
        xxs=np.array([szi,sxi]).reshape(1,2)
        XXs = np.concatenate((XXs,xxs),0)

        input = vel[np.newaxis,np.newaxis, :]

        vs = vel[int(round(szi / deltaz)), int(round(sxi / deltax))]
        phi = -1 * np.ones_like(X)
        phi[int(szi/deltax)][int(sxi/deltax)] = 1
        d = skfmm.distance(phi, dx=2e-2)
        T_data = skfmm.travel_time(phi, vel, dx=2e-2)

        T0 = np.sqrt((Z - szi) ** 2 + (X - sxi) ** 2) / vs
        tau = np.divide(T_data, T0, out=np.ones_like(T0), where=T0 != 0)
        tau = tau[None, None, :, :]

        input_data = np.concatenate((input_data, input), 0)
        tau_data = np.concatenate((tau_data, tau),0)

print("input_data.shape:",input_data.shape)
print("XXs.shape.shape:",XXs.shape)
print("tau_data.shape:",tau_data.shape)

np.savez('input_valid', input_data=input_data,XXs=XXs)
np.savez('target_valid', tau_data=tau_data)

