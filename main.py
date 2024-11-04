import warnings

import matplotlib.pyplot as plt

plt.ion()
import numpy as np

import pysindy as ps

# ignore user warnings
warnings.filterwarnings("ignore", category=UserWarning)

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

from utils import (
    compare_methods,
    print_equations,
    compare_coefficient_plots,
    plot_sho,
    plot_lorenz,
)

diffs = [
    ("PySINDy Finite Difference", ps.FiniteDifference()),
    ("Finite Difference", ps.SINDyDerivative(kind="finite_difference", k=1)),
    ("Smoothed Finite Difference", ps.SmoothedFiniteDifference()),
    (
        "Savitzky Golay",
        ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3),
    ),
    ("Spline", ps.SINDyDerivative(kind="spline", s=1e-2)),
    ("Trend Filtered", ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=1e-2)),
    ("Spectral", ps.SINDyDerivative(kind="spectral")),
    ("Spectral, PySINDy version", ps.SpectralDerivative()),
    ("Kalman", ps.SINDyDerivative(kind="kalman", alpha=0.05)),
]

noise_level = 0.01
# True data
x, y, y_noisy, y_dot = gen_data_sine(noise_level)
axs = compare_methods(diffs, x, y, y_noisy, y_dot)
plt.show()


# Shrink window for Savitzky Golay method
diffs[3] = (
    "Savitzky Golay",
    ps.SINDyDerivative(kind="savitzky_golay", left=0.1, right=0.1, order=3),
)
diffs[8] = ("Kalman", ps.SINDyDerivative(kind="kalman", alpha=0.01))

x, y, y_dot, y_noisy = gen_data_step(noise_level)

axs = compare_methods(diffs, x, y, y_noisy, y_dot)
plt.show()

figure = plt.figure(figsize=[5, 5])
plot_sho(x_train, x_train_noisy)

# Allow Trend Filtered method to work with linear functions
diffs[5] = (
    "Trend Filtered",
    ps.SINDyDerivative(kind="trend_filtered", order=1, alpha=1e-2),
)
diffs[8] = ("Kalman", ps.SINDyDerivative(kind="kalman", alpha=0.5))
diffs.append(("Smooth FD, reuse old x", ps.SmoothedFiniteDifference(save_smooth=False)))
diffs.append(
    (
        "Kalman, reuse old x",
        ps.SINDyDerivative(kind="kalman", alpha=0.5, save_smooth=False),
    )
)

equations_clean = {}
equations_noisy = {}
coefficients_clean = {}
coefficients_noisy = {}
input_features = ["x", "y"]
threshold = 0.5

for name, method in diffs:
    model = ps.SINDy(
        differentiation_method=method,
        optimizer=ps.STLSQ(threshold=threshold),
        t_default=dt,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True)
    equations_clean[name] = model.equations()
    coefficients_clean[name] = model.coefficients()

    model.fit(x_train_noisy, quiet=True)
    equations_noisy[name] = model.equations()
    coefficients_noisy[name] = model.coefficients()