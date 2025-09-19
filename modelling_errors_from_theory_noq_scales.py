# Standard
from functools import partial
from tqdm import tqdm

# Third Party
from scipy.integrate import quad, trapezoid
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


# ============================================
# Customization parameters
BLOCK_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
MAX_INTEGRATION_POINTS = 100
# ============================================


# Range of standard deviation sigma to model
sigmas = np.linspace(0.001, 0.050, 50)
sigmas = np.insert(sigmas, 0, 0.0001)

# Quantization level and Voronoi bin edges for FP4 E2M1 (elements quantization)
# Probability density outside [-6, 6] is zero --> edges are truncated
qlevels = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
voronoi_edges = [-6.0, -5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
assert len(qlevels) == len(voronoi_edges) - 1

# Define Standard Normal PDF and CDF (mu = 0, std = 1)
phi = norm.pdf
Phi = norm.cdf


def f_xmax(z: np.float64, sigma: np.float64, N: int) -> np.float64:
    """Compute PDF of the maximum of N i.i.d. N(0, sigma^2) variables.
    Obtained from the derivative of CDF = [2 * Phi(z / sigma) - 1]^N
    Can be expressed in terms of error function erf or equivalently as Phi (the CDF)
    N is selected block size (bs)
    """

    return 2 * N / sigma * (2 * Phi(z / sigma) - 1)**(N-1) * phi(z/sigma)


def conditional_error(z: np.float64, sigma: np.float64, N:int) -> np.float64:
    """Compute quantization error conditioned on x_max"""

    alpha = z / (6 * sigma)

    norm_const = Phi(6 * alpha) - Phi(-6 * alpha)
    # NOTE: two alternative formulations for norm_const (identical output):
    #   2*Phi(6*alpha) - 1
    #   erf(6*alpha/np.sqrt(2))

    mse_x_total = 0.0
    for i in range(len(qlevels)):
        # Quantization level and scaled Voronoi boundaries of microscaling elements
        q = qlevels[i]
        v = voronoi_edges[i] * alpha
        w = voronoi_edges[i + 1] * alpha

        # Integrate error square time probability density to get quantization error
        e_bin_prenorm, _ = quad(
            lambda u: ((u - q * alpha) ** 2) * phi(u),
            v,
            w,
            limit=MAX_INTEGRATION_POINTS,
        )
        e_bin = sigma**2 / norm_const * e_bin_prenorm
        mse_x_total += e_bin

    return mse_x_total * (N-1) / N


# Compute MSE_Z for each block size and sigma
mse = {bs:np.zeros(len(sigmas)) for bs in BLOCK_SIZES}
for bs in BLOCK_SIZES:
    for j, sigma in enumerate(tqdm(sigmas, desc=f"{bs=:3}")):
        z_vals = np.linspace(0.001, 8 * sigma, 200)  # discretization of x_max
        integrand_vals = np.zeros(len(z_vals))
        for i, z in enumerate(z_vals):
            err = conditional_error(z, sigma, bs)
            fz = f_xmax(z, sigma, bs)
            integrand_vals[i] = err * fz
        mse[bs][j] = trapezoid(integrand_vals, z_vals)


# Save all results
bs_str = '-'.join(str(k) for k in BLOCK_SIZES)
np.save(
    f"data_theory_noq_scales_bs{bs_str}.npy",
    {"sigma": sigmas, "mse": mse},
)
