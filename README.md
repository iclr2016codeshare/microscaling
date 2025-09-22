# Is Finer Better? The Limits of Microscaling Formats in Large Language Models

Code shared anonimously as part of ICLR 2016 submission process.

This repo builds on [microxcaling](https://github.com/microsoft/microxcaling) and [MX-QLLM](https://github.com/aiha-lab/MX-QLLM), adding UE5M3 functionalities. We primarily modify `microxcaling/mx/linear.py` and `microxcaling/mx/cpp/mx.cuh` to add this support.

Install as per original instructions.

Experiments were run using:
```
torch                     2.7.1+cu128
transformers              4.56.1
lm_eval                   0.4.9.1
mamba_ssm                 2.2.5
causal_conv1d             1.5.2
fast_hadamard_transform   1.0.4.post1
```

We also provide python scripts to model errors in ideal distributions and using our theoretical framework, with FP8 scales or not quantized.
