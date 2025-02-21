import numpy as np
import skfmm
from scipy.ndimage import gaussian_filter1d

data = np.load('velmodels_train_read.npz')
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

# print(Z.shape)
Zra, Xra, Zsa, Xsa, Vsa, Tref, tauref, = [], [], [], [], [], [], []

input_data = np.empty((0,1,70,70))
tau_data = np.empty((0,1,70,70))
XXs = np.empty((0,2))
flag = 0
for i in range(0, len(velmodels)):

    selected_pts1 = np.random.choice(a=10, size=1,replace=False)
    selected_pts2 = np.random.choice(a=len(Z) - 1, size=4, replace=False)
    Zr = Z
    Xr = X

    start = selected_pts1[0]*10

    topo = np.loadtxt('elevation.txt')[start:start + 140:2, 2] / 1000
    topo = gaussian_filter1d(topo, 4)
    topo = (100 * np.gradient(np.gradient(topo)) + np.round(1.2 + 1.2 * np.sin(x) * np.cos(x), 4)) / 10

    Xs = X[0][selected_pts2]
    Zs = [int((topo[int(Xs[i] / deltax)] + deltax)/deltax)*deltax for i in range(len(Xs))]

    # Creating grid with points above topography markes as NaN
    Z_ = []
    X_ = []
    # For each x component, the loop marks y values above topo at that x to be Nan
    for j in enumerate(x):
        index = j[0];
        xval = j[1]
        ztemp = [z[j[0]] if z[j[0]] >= np.floor(topo[index] * 100) / 100 else float("Nan") for j in enumerate(z)]
        Z_ = np.append(Z_, ztemp)
        X_ = np.append(X_, np.ones(len(ztemp)) * xval)
    # Copying Nan from Z to X
    X_ = X_ + Z_ * 0.
    # Reshaping X and Z to the original model size
    X_ = X_.reshape(np.meshgrid(z, x, indexing='ij')[0].shape).T
    Z_ = Z_.reshape(np.meshgrid(z, x, indexing='ij')[0].shape).T
    TOPO = np.divide(X_, X_, out=np.ones_like(X_), where=X_ != 0)
    TOPO0 = np.nan_to_num(TOPO, nan=0)

    vel = velmodels[i, :, :] * TOPO0

    for ns, (szi, sxi) in enumerate(zip(Zs, Xs)):
        xxs=np.array([szi,sxi]).reshape(1,2)
        XXs = np.concatenate((XXs,xxs),0)

        input = np.concatenate((vel[np.newaxis, :],TOPO0[np.newaxis, :]), 0)
        input = input[None, :, :, :]

        vs = vel[int(round(szi / deltaz)+1), int(round(sxi / deltax))]
        mask = (vel == 0)
        phi = -1 * np.ones_like(X)
        phi[int(szi/deltax)+1][int(sxi/deltax)] = 0
        phi = np.ma.MaskedArray(phi, mask)

        d = skfmm.distance(phi, dx=2e-2)
        T_data = skfmm.travel_time(phi, vel, dx=2e-2)

        T0 = np.sqrt((Z - szi) ** 2 + (X - sxi) ** 2) / vs
        tau = np.divide(T_data, T0, out=np.ones_like(T0), where=T0 != 0)
        tau = tau[None, None, :, :]
        if flag == 0:
            input_data = input
            tau_data = tau
            flag = 1
        else:
            input_data = np.concatenate((input_data, input), 0)
            tau_data = np.concatenate((tau_data, tau),0)

print("input_data.shape:",input_data.shape)
print("tau_data.shape:",tau_data.shape)
print("XXs.shape:",XXs.shape)

np.savez('input_train', input_data=input_data,XXs=XXs)
np.savez('target_train', tau_data=tau_data)

