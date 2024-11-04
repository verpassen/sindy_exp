import numpy as np 
import matplotlib as plt 
import pysindy as ps 
from pysindy.feature_library import PDELibrary

# Burgers’ equation


# Load data from .mat file
b_show = False
data = np.loadtxt('burgers_full.txt')
# t = np.ravel(data['t'])
# x = np.ravel(data['x'])
t,x = np.arange(0,40) , np.arange(41)
u = np.real(data)
dt = t[1] - t[0]
dx = x[1] - x[0]
print(dt)
if b_show:
    # Plot u and u_dot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, x, u)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    plt.title(r'$u(x, t)$', fontsize=16)

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, x, u_dot)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.title(r'$\dot{u}(x, t)$', fontsize=16)
    plt.show()

u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
u = u.reshape(len(x), len(t), 1)
u_dot = u_dot.reshape(len(x), len(t), 1)


# library_functions = [lambda x: x, lambda x: x * x]
# library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.PDELibrary(
    # library_functions=library_functions,
    # function_names=library_function_names,
    library_functions=ps.PolynomialLibrary(degree=2,include_bias=False),
    derivative_order=3,
    spatial_grid=x,
    is_uniform=True,
)

print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

print('SR3 model, L0 norm: ')
optimizer = ps.SR3(
    threshold=2,
    max_iter=10000,
    tol=1e-15,
    nu=1e2,
    thresholder="l0",
    normalize_columns=True,
)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(np.real(data), t=dt)
model.print()

print('SR3 model, L1 norm: ')
optimizer = ps.SR3(
    threshold=0.5, max_iter=10000, tol=1e-15,
    thresholder="l1", normalize_columns=True
)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

print('SSR model: ')
optimizer = ps.SSR(normalize_columns=True, kappa=1)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

print('SSR (metric = model residual) model: ')
optimizer = ps.SSR(criteria="model_residual",
                   normalize_columns=True,
                   kappa=1)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

print('FROLs model: ')
optimizer = ps.FROLS(normalize_columns=True, kappa=1e-3)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()