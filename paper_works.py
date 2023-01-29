import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import numpy as np

np.set_printoptions(precision=3)
from scipy.optimize import curve_fit
from scipy.optimize import least_squares


### first guess for the shape of the data when plotted against x
def my_p(x, a, p):
    return a * x ** p


### turns out that the fit parameters theselve follow the above law
### themselve but with an offset and plotted against z, of course
def my_p2(x, a, p, x0):
    return a * (x - x0) ** p


def my_full(x, y, asc, aexp, ash, psc, pexp, psh):
    a = my_p2(y, asc, aexp, ash)
    p = my_p2(y, psc, pexp, psh)
    z = my_p(x, a, p)
    return z


### base lists for plotting
xl = np.linspace(0, 30, 150)
yl = np.linspace(5, 200, 200)

### setting the plots
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(2, 3, 1)
bx = fig.add_subplot(2, 3, 2)
cx = fig.add_subplot(2, 3, 3)
dx = fig.add_subplot(2, 3, 4)
ex = fig.add_subplot(2, 3, 5)
fx = fig.add_subplot(2, 3, 6, projection='3d')

### fitting the data linewise for different z as function of x
### keeping track of the fit parameters
adata = list()
pdata = list()
for subdata in data:
    xdata = subdata[::, 1]
    zdata = subdata[::, 3]
    sol, _ = curve_fit(my_p, xdata, zdata)
    print(sol, subdata[0, 2])  ### fit parameters for different z
    adata.append([subdata[0, 2], sol[0]])
    pdata.append([subdata[0, 2], sol[1]])
    ### plotting the z-cuts
    ax.plot(xdata, zdata, ls='', marker='o')
    ax.plot(xl, my_p(xl, *sol))

adata = np.array(adata)
pdata = np.array(pdata)

ax.scatter([0], [0])
ax.grid()

### fitting the the fitparameters as function of z
sola, _ = curve_fit(my_p2, adata[::, 0], adata[::, 1], p0=(1, -0.05, 0))
print(sola)
bx.plot(*(adata.transpose()))
bx.plot(yl, my_p2(yl, *sola))

solp, _ = curve_fit(my_p2, pdata[::, 0], pdata[::, 1], p0=(1, -0.05, 0))
print(solp)
cx.plot(*(pdata.transpose()))
cx.plot(yl, my_p2(yl, *solp))

### plotting the cuts applying the resuts from the "fit of fits"
for subdata in data:
    xdata = subdata[::, 1]
    y0 = subdata[0, 2]
    zdata = subdata[::, 3]
    dx.plot(xdata, zdata, ls='', marker='o')
    dx.plot(
        xl,
        my_full(
            xl, y0, 2.12478827, -0.187, -20.84, 0.928, -0.0468, 0.678
        )
    )


### now fitting the entire data with the overall 6 parameter function
def residuals(params, alldata):
    asc, aexp, ash, psc, pexp, psh = params
    diff = list()
    for data in alldata:
        x = data[1]
        y = data[2]
        z = data[3]
        zth = my_full(x, y, asc, aexp, ash, psc, pexp, psh)
        diff.append(z - zth)
    return diff


## and fitting using the hand-made residual function and least_squares
resultfinal = least_squares(residuals,x0=(2.12478827, -0.187, -20.84, 0.928, -0.0468, 0.678), args=(data0,))

### least_squares does not provide errors but the approximated jacobian
### so we follow:
### https://stackoverflow.com/q/61459040/803359
### https://stackoverflow.com/q/14854339/803359
print(resultfinal.x)
resi = resultfinal.fun
JMX = resultfinal.jac
HMX = np.dot(JMX.transpose(), JMX)
cov_red = np.linalg.inv(HMX)
s_sq = np.sum(resi ** 2) / (len(data0) - 6)
cov = cov_red * s_sq

print(cov)

### plotting the cuts with the overall fit
for subdata in data:
    xdata = subdata[::, 1]
    y0 = subdata[0, 2]
    zdata = subdata[::, 3]
    ex.plot(xdata, zdata, ls='', marker='o')
    ex.plot(xl, my_full(xl, y0, *resultfinal.x))

### and in 3d, which is actually not very helpful partially due to the
### fact that matplotlib has limited 3d capabilities.
XX, YY = np.meshgrid(xl, yl)
ZZ = my_full(XX, YY, *resultfinal.x)
fx.scatter(
    data0[::, 1], data0[::, 2], data0[::, 3],
    color="#ff0000", alpha=1)
fx.plot_wireframe(XX, YY, ZZ, cmap='inferno')

plt.show()
