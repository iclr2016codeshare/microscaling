# Standard
from tqdm import tqdm

# Third Party
from scipy.integrate import quad, trapezoid
from scipy.special import erf
from scipy.stats import norm
import numpy as np


# ============================================
# Customization parameters
BLOCK_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
MAX_INTEGRATION_POINTS = 100
NUM_SIGMAS = 50
# ============================================

# Range of standard deviation sigma to model
sigmas = np.linspace(0.001, 0.050, NUM_SIGMAS)
sigmas = np.insert(sigmas, 0, 0.0001)

# Quantization level and Voronoi bin edges for FP4 E2M1 (elements quantization)
# Probability density outside [-6, 6] is zero --> edges are truncated
element_qlevels = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
element_qedges = [-6.0, -5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
assert len(element_qlevels) == len(element_qedges) - 1, "Incorrect FP4 levels derivation."

# NOTE: two equivalent derivations of FP4 edges
# element_qedges = [element_qlevels[0]] + [(element_qlevels[i] + element_qlevels[i+1])/2 for i in range(len(element_qlevels) - 1)] + [element_qlevels[-1]]
# element_qedges = np.concatenate([[element_qlevels[0]], (element_qlevels[1:] + element_qlevels[:-1]) / 2, [element_qlevels[-1]]])


def get_positive_scale_qlevels(e: int = 4, m: int = 3) -> np.array:
    """Define FP8 quantization levels.
    Default: FP8 E4M3
    """

    qlevels = np.array([])
    bias = 2**(e-1) - 1
    # Subnormals numbers
    for i in range(2**m):  # [0,7]
        qlevels = np.append(qlevels, (i + 1) / 2**m * 2**(1 - bias))
    # Normal numbers
    for j in range(1, 2**e):  # [1, 15]
        for i in range(2**m):  # [0, 7]
            if j == (2**e - 1) and i == 2**m - 1:
                break   # skip inf (480 for fp8e4m3)
            qlevels = np.append(qlevels, 2**(j - bias) * (1 + i/2**m))
    return qlevels

# Quantization level and Voronoi bin edges for FP8 E4M3 (scales quantization)
scale_qlevels = get_positive_scale_qlevels()
scale_qedges = np.concatenate([[scale_qlevels[0] / 2], (scale_qlevels[1:] + scale_qlevels[:-1])/2, [scale_qlevels[-1]]])
assert len(scale_qlevels) == len(scale_qedges) - 1, "Incorrect FP8 levels derivation."

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


def quantize_y(y: np.float64, element_qlevels: list) -> float:
    """Return quantized input after elements quantization"""

    return element_qlevels[np.argmin(np.abs(y - element_qlevels))]


def probability_s_i(a: np.float64, b: np.float64, sigma: np.float64, bs: int) -> np.float64:
    """Single FP8 scale bin probability mass"""

    f_sx = lambda s: f_xmax(6 * s, sigma=sigma, N=bs) * 6
    prob, _ = quad(f_sx, a, b, limit=MAX_INTEGRATION_POINTS)
    return prob


def conditional_error_zero_scale(zero_scale_edge: np.float64, sigma: np.float64, N: int) -> np.float64:
    """Compute quantization error for the scale bin with s=0"""

    if sigma <= 0 or zero_scale_edge <= 0:
        return 0.0

    alpha_x = 6 * zero_scale_edge / sigma

    prob_one_x_rounds_to_zero = 2 * Phi(alpha_x) - 1
    prob_zero_scale = prob_one_x_rounds_to_zero**N  # probability N variables are < zero_scale_edge
    conditional_expected_x_squared = sigma**2 * (1 - (2 * alpha_x * phi(alpha_x)) / prob_one_x_rounds_to_zero)
    msez_s_i_zero = prob_zero_scale * conditional_expected_x_squared

    return msez_s_i_zero


def conditional_error_xmax(s: np.float64, a: np.float64, b: np.float64, sigma: np.float64, N: int, element_qlevels: list) -> np.float64:
    """Compute quantization error for x_i = x_max at given scale s"""

    e_bin_xmax_prenorm, _ = quad(
        lambda x: (quantize_y(x / s, element_qlevels) * s - x)**2 * f_xmax(x, sigma=sigma, N=N),
        6*a,
        6*b,
        limit=MAX_INTEGRATION_POINTS,
    )
    p_xi, _ = quad(
        lambda x: f_xmax(x, sigma=sigma, N=N),
        6*a,
        6*b,
        limit=MAX_INTEGRATION_POINTS,
    )
    if p_xi == 0:
        return 0.0

    msez_per_si_xmax = e_bin_xmax_prenorm / (N * p_xi)

    return msez_per_si_xmax


def conditional_error_not_xmax(s: np.float64, sigma: np.float64, N: int) -> np.float64:
    """Compute quantization error for x_i != x_max at given scale s"""

    alpha = s / sigma

    norm_const = Phi(6 * alpha) - Phi(-6 * alpha)
    # NOTE: two alternative formulations for normalization constant (identical output):
    #   2*Phi(6*alpha) - 1
    #   erf(6*alpha/np.sqrt(2))

    msez_per_si = 0.0
    for i in range(len(element_qlevels)):
        # Quantization level and scaled Voronoi boundaries of microscaling elements
        q = element_qlevels[i]
        v = element_qedges[i] * alpha
        w = element_qedges[i + 1] * alpha

        # Quantization error for x_i != x_max
        e_bin_prenorm, _ = quad(
            lambda u: ((u - q * alpha) ** 2) * phi(u),
            v,
            w,
            limit=MAX_INTEGRATION_POINTS,
        )
        e_bin = sigma**2 / norm_const * e_bin_prenorm * (N-1) / N
        msez_per_si += e_bin

    return msez_per_si


# Compute MSE_Z for each block size and sigma
msez = {bs:np.zeros(len(sigmas)) for bs in BLOCK_SIZES}
msez_per_si_xmax_all = {bs:np.zeros(len(sigmas)) for bs in BLOCK_SIZES}
msez_per_si_not_xmax_prescaling_all = {bs:np.zeros(len(sigmas)) for bs in BLOCK_SIZES}
msez_per_si_zero = {bs:np.zeros(len(sigmas)) for bs in BLOCK_SIZES}
for bs in BLOCK_SIZES:
    for j, sigma in enumerate(tqdm(sigmas, desc=f"{bs=:3}")):
        for i in range(len(scale_qlevels)):
            s_i = scale_qlevels[i]
            a_i = scale_qedges[i]
            b_i = scale_qedges[i+1]

            # Error source #1: x_1 = x_max
            msez_per_si_xmax = conditional_error_xmax(s_i, a_i, b_i, sigma, bs, element_qlevels)

            # Error source #2: x_i != x_max
            msez_per_si_not_xmax_prescaling = conditional_error_not_xmax(s_i, sigma, bs)

            # Probability mass of scale s_i
            p_i = probability_s_i(a_i, b_i, sigma, bs)

            # Compute total MSE_Z for scale s_i
            msez_per_si = p_i * (msez_per_si_not_xmax_prescaling + msez_per_si_xmax)
            msez_per_si_xmax_all[bs][j] += p_i * msez_per_si_xmax
            msez_per_si_not_xmax_prescaling_all[bs][j] += p_i * msez_per_si_not_xmax_prescaling
            msez[bs][j] += msez_per_si

        # Error source #3: s=0
        msez_per_si_zero[bs][j] = conditional_error_zero_scale(scale_qedges[0], sigma, bs)

        # MSE_Z at given BS and SIGMA
        msez[bs][j] += msez_per_si_zero[bs][j]


# Save all results
bs_str = '-'.join(str(k) for k in BLOCK_SIZES)
np.save(
    f"data_theory_fp8_scales_bs{bs_str}.npy",
    {
        "sigma": sigmas,
        "mse": msez,
        "msez_zero_scale": msez_per_si_zero,
        "mse_xmax": msez_per_si_xmax_all,
        "mse_not_xmax": msez_per_si_not_xmax_prescaling_all,
    },
)
