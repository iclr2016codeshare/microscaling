"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')
import math

f_linear = F.linear
torch_matmul = torch.matmul

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        mx_specs=None,
        name=None,
        args=None,
    ):
        dtype = input.dtype
        # element-wise quantize for input
        bf_in = quantize_elemwise_op(
            deepcopy(input.float()), mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            deepcopy(weight.float()), mx_specs=mx_specs, round=mx_specs["round_weight"]
        )

        if bias is not None:
            ctx.has_bias = True
            bf_bias = quantize_elemwise_op(
                bias.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
            ).to(dtype)
        else:
            ctx.has_bias = False
            bf_bias = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        # ============================================
        if mx_specs["pertensor_wscale"] or mx_specs["pertensor_ascale"]:
            if mx_specs['w_elem_format'] == "fp4_e2m1":
                elem_max_norm = 6 # 2**(2-1)*(1+0.5)
            else:
                elem_max_norm = 1
            if mx_specs['w_scale_mode'] == 143:
                scale_max_norm = 480  # 2**(15-7)*(1+7/8); no NaN/inf
            elif mx_specs['w_scale_mode'] == 152:
                scale_max_norm = 57344 # 2**(30-15)*(1+3/4); exp=11111 reserved for inf
            elif mx_specs['w_scale_mode'] == 53:
                scale_max_norm = 61440  # 2**(30-15)*(1+7/8); exp=11111 reserved for inf
            elif mx_specs['w_scale_mode'] == 0 and mx_specs['e8m0_scale'] > 0:
                scale_max_norm = mx_specs['e8m0_scale']
            else:
                scale_max_norm = 1

        if mx_specs["pertensor_ascale"]:
            activ_tensor_scale = (elem_max_norm * scale_max_norm) / bf_in.abs().max()
            if activ_tensor_scale > 1e10 or activ_tensor_scale < 1:
                activ_tensor_scale = torch.tensor([1]).to(bf_in.device)
            bf_in *= activ_tensor_scale

        # MX quantize everything along input size
        qis_input = quantize_mx_op(  # replace with qis_input_scaled if scaling A
            bf_in,
            mx_specs,
            elem_format=mx_specs['a_elem_format'],
            scale_mode=mx_specs['a_scale_mode'],
            scale_min=mx_specs["scale_min"],
            axes=[-1],  # quantize along the embedding_feature dimension (= w_in_feat)
            round=mx_specs["round_mx_output"],
        )

        if mx_specs["pertensor_wscale"]:
            weight_tensor_scale = (elem_max_norm * scale_max_norm) / bf_weight.abs().max()
            if weight_tensor_scale > 1e10 or weight_tensor_scale < 1:
                weight_tensor_scale = torch.tensor([1]).to(bf_weight.device)
            bf_weight *= weight_tensor_scale

        qis_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            scale_mode=mx_specs['w_scale_mode'],
            scale_min=mx_specs["scale_min"],
            axes=[-1],  # quantize along the in_feat dimension (= x_emb_dim)
            round=mx_specs["round_mx_output"],
        )

        # Scale back weights and/or activations
        if mx_specs["pertensor_wscale"]:
            qis_weight /= weight_tensor_scale
        if mx_specs["pertensor_ascale"]:
            qis_input /= activ_tensor_scale

        qis_input = qis_input.to(dtype)
        qis_weight = qis_weight.to(dtype)

        # compute output
        output = f_linear(qis_input, qis_weight)

        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_weight, grad_bias
        #####################################################
        # if the input is 2D, quantize everything along examples (batches)
        # if the input is 3D, quantize everything along the first axis
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_weight"],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_weight"],
        )

        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # Compute grad_weight
        grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # perform madtile operation for grad_input
        #####################################################
        # compute grad_input, quantize everything along output size
        qos_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format_bp'],
            axes=[0],
            round=ctx.mx_specs["round_mx_weight_grad_input"],
        )
        # grad_output shape is (B, seq, out_dim)
        qos_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_os'],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # Compute grad_input
        grad_input = torch_matmul(qos_grad_output, qos_weight)
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    mx_specs=None,
    name=None,
    args=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    return LinearFunction.apply(input, weight, bias, mx_specs, name, args)


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        args=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        super().__init__(in_features, out_features, bias)
        self.args = args

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)

        # Hadamard transform (QuaRot)
        if hasattr(self, "online_full_had"):
            inputs = hadamard_utils.matmul_hadU_cuda(inputs, self.had_K, self.K)
        elif hasattr(self, "online_partial_had"):
            init_shape = inputs.shape
            if self.K == 1:
                inputs = fast_hadamard_transform.hadamard_transform(
                    inputs.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1/math.sqrt(init_shape[-1]//self.had_dim)
                ).transpose(1, 2)
            else:
                inputs = (self.had_K.to(inputs.dtype).to(inputs.device) @ inputs.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
            inputs = inputs.reshape(init_shape)

        out = linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            name=self.name,
            args=self.args,
        )

        return out
