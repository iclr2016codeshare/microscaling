# Standard
from copy import deepcopy
from tqdm import tqdm

# Third Party
from torch.distributions import Gumbel, Laplace, Normal, Uniform, VonMises
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import numpy as np
import torch

# Local
from mx.mx_ops import quantize_mx_op
from mx.elemwise_ops import quantize_elemwise_op


# ============================================
# Customization parameters
DEFAULT_DTYPE = torch.bfloat16
SEED = 83013
BLOCK_SIZES = [2, 4] #, 8, 16, 32, 64, 128, 256]
SCALE_MODE = 143  # 143: UE4M3; 53: UE5M3; 16: BF16
NUM_SIGMAS = 50
SAMPLE_ROWS = 1024
SAMPLE_COLS = 1024
# ============================================


torch.manual_seed(SEED)
torch.set_default_dtype(DEFAULT_DTYPE)
sigmas = np.linspace(0.001, 0.050, 199)
sigmas = np.insert(sigmas, 0, 0.0001)

# Initialize microscaling configuration (will be partially overwritten later)
mx_specs = {
    'scale_bits': 8,
    'w_elem_format': 'fp4_e2m1',
    'a_elem_format': 'fp4_e2m1',
    'A_elem_format': None,
    'B_elem_format': None,
    'w_elem_format_bp': None,
    'a_elem_format_bp_ex': None,
    'a_elem_format_bp_os': None,
    'mx_flush_fp32_subnorms': False,
    'shared_exp_method': 'max',
    'block_size': 8,
    'bfloat': 0,
    'fp': 0,
    'bfloat_subnorms': True,
    'quantize_backprop': True,
    'round': 'nearest',
    'round_m': 'nearest',
    'round_weight': 'nearest',
    'round_output': 'nearest',
    'round_grad_weight': 'nearest',
    'round_grad_input': 'nearest',
    'round_mx_output': 'nearest',
    'round_mx_input_grad_input': 'nearest',
    'round_mx_weight_grad_input': 'nearest',
    'round_mx_grad_output_grad_input': 'nearest',
    'round_mx_input_grad_weight': 'nearest',
    'round_mx_grad_output_grad_weight': 'nearest',
    'softmax_exp2': False,
    'vec_use_exp2': False,
    'vec_use_recip': False,
    'custom_cuda': True,
    'a_scale_mode': 143,
    'w_scale_mode': 143,
    'A_scale_mode': 0,
    'B_scale_mode': 0,
    'per_tensor': False,
    'e8m0_scale': -1.0,
    'pertensor_wscale': False,
    'pertensor_ascale': False,
    'scale_min': -1,
}

# NOTE: using FP64 precision to extend limits of RNG process that uses inverse transform
DISTRIBUTIONS = {
    "normal": Normal(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64)),
    "laplace": Laplace(torch.tensor(0, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)),
    "vonmises": VonMises(torch.tensor(0, dtype=torch.float64), torch.tensor(1.6, dtype=torch.float64)),
    "logistic": (
        TransformedDistribution(
            Uniform(torch.tensor(0, dtype=torch.float64), torch.tensor(1, dtype=torch.float64)),
            [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
        )
    ),
    "gumbel": Gumbel(torch.tensor(0, dtype=torch.float64), torch.tensor(0.8, dtype=torch.float64)),
}

# Begin processing
mse = {}
std = {}
std_per_block  = {}
for bs in BLOCK_SIZES:
    print(f"Processing block size {bs}")
    mse[bs] = {}
    std[bs] = {}
    for name, dist in DISTRIBUTIONS.items():
        if name not in mse[bs]:
            mse[bs][name] = {}
            std[bs][name] = {}
        mse[bs][name] = np.zeros(len(sigmas))
        std[bs][name] = np.zeros(len(sigmas))
        for i, sigma in tqdm(enumerate(sigmas), desc=name):
            # Draw SAMPLE_ROWS*SAMPLE_COLS samples from selected distribution
            w = dist.sample((SAMPLE_ROWS, SAMPLE_COLS)).cuda().to(torch.float32) * sigma

            # convert weights to BF16
            w_bf = quantize_elemwise_op(
                w, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )

            # Quantize tensor with selected FP4 microscaling format
            mx_specs["block_size"] = bs
            mx_specs['w_scale_mode'] = SCALE_MODE
            wq = quantize_mx_op(
                deepcopy(w_bf),
                mx_specs,
                elem_format=mx_specs['w_elem_format'],
                scale_mode=mx_specs['w_scale_mode'],
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )

            # Compute MSE_Z and standard deviation at tensor level
            mse[bs][name][i] = ((wq - w_bf)**2).mean().item()
            std[bs][name][i] = w_bf.std()

bs_str = '-'.join(str(k) for k in BLOCK_SIZES)
file_name = f"data_ideal_distrib_scale-{SCALE_MODE}_bs{bs_str}_TEST.npy"
np.save(file_name, {"sigma":std, "mse":mse})
print(f"Data saved to file: {file_name}")
