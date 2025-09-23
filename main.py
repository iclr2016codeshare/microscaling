# Adapted from https://github.com/aiha-lab/MX-QLLM/blob/main/main.py

# Standard
import argparse
import logging
import os
import warnings

# Third Party
from accelerate import Accelerator
import lm_eval
from lm_eval import evaluator
import torch
from torch import nn
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM

# Local
from utils.common import set_seed, str2bool, str2int, str2list
from utils.mx import get_mx_model, parse_mx_specs

warnings.filterwarnings('ignore')

SUPPORTED_MODELS = [
    "granite-3.0",
    "granite-3.1",
    "granite-3.2",
    "granite-3.3",
    "llama-4",
    "llama-3.1",
    "llama-2",
    "llama",
    "gpt-oss",
    "mixtral-8x7b",
    "qwen3",
    "qwen2",
    "qwen",
    "bamba",  # mamba2 hybrid
    "mamba2",
    "codestral",  # mamba2
    "falcon3",  # mamba1
    "zamba2",  # mamba2
    "hymba",  # NVIDIA mamba2
    "gpt",
    "opt",
    "nemotron",  # NVIDIA mamba2
]


def main(args):
    """Main function to apply MX quantization and run evaluation."""

    print("=" * 60)
    print(f"CONDA_DEFAULT_ENV = {os.environ.get('CONDA_DEFAULT_ENV', 'CONDA_DEFAULT_ENV not found')}")
    print(f"CONDA_PREFIX = {os.environ.get('CONDA_PREFIX', 'CONDA_PREFIX not found')}")
    print("CURRENT MAMBA ENV:", flush=True)
    _ = os.system("mamba env list | grep '*'")
    print("=" * 60)

    #============================ Load model
    if args.quarot: # QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)
        logging.info("Applying Hadamard transform following QuaRot")
        kwargs = {'device_map':'cpu','trust_remote_code':True,'attn_implementation':"eager"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype='auto' if args.auto_dtype else torch.float32,
            **kwargs,
        )
        from scale_utils import rotation_utils
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        rotation_utils.cleanup_memory(verbos=True)
    else:
        dev = "balanced"  # balanced  sequential
        kwargs = {'device_map':dev,'trust_remote_code':True,'attn_implementation':"eager"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype='auto' if args.auto_dtype else torch.float32,
            **kwargs,
        )

    # load fast tokenizer for selected models
    use_fast_tok = True if any(
        k in args.model for k in [
            "zamba2", "granite-3.", "mamba2", "llama-3", "llama-4", "nemotron"
        ]
    ) else False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=use_fast_tok,
    )

    # ensure model is fully loaded on GPU (or CPU)
    assert not any(v.device == torch.device("meta") for v in model.parameters())

    #============================ MX format
    mx_specs_linear=parse_mx_specs(args,'linear')
    mx_specs_matmul=parse_mx_specs(args,'matmul')
    mx_specs_ln=parse_mx_specs(args,'ln')
    mx_specs_head=parse_mx_specs(args,'head')

    # NOTE: modify this to exclude selected linear layers from quantization
    exclude_keys = ["adapter"]  # never quantize LoRA adapters, if present

    get_mx_model(
        model.eval(),
        mx_specs_linear=mx_specs_linear,
        mx_specs_matmul=mx_specs_matmul,
        mx_specs_ln=mx_specs_ln,
        mx_specs_head=mx_specs_head,
        args=args,
        exclude_keys=exclude_keys,
    )

    print(model)
    model_iter = iter(model.parameters())
    _ = next(model_iter) # skip first parameter (embedding)
    model_dtype = next(model_iter).dtype
    print(f"DATATYPE = {model_dtype}")
    print("=" * 60)


    #============================ Runtime Hadamard Transform for QuaRot
    if args.quarot:
        from scale_utils import hadamard_utils
        if 'llama' in args.model or 'mistral' in args.model:
            for name, module in model.named_modules():
                if 'down_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    setattr(module, "online_full_had", True)
                    setattr(module, "had_K", had_K)
                    setattr(module, "K", K)
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    setattr(module, "online_partial_had", True)
                    setattr(module, "had_K", had_K)
                    setattr(module, "K", K)
                    setattr(
                        module,
                        "had_dim",
                        model.config.hidden_size//model.config.num_attention_heads,
                    )
        else:
            raise NotImplementedError

        # KV Cache
        if args.rotate_kv:
            from scale_utils import model_utils, rotation_utils
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn,
                            rope_function_name,
                            config=model.config,
                )

    # Load into GPU
    if model.device.type=='cpu':
        accelerator = Accelerator()
        model = accelerator.prepare(model)

    #============================ Evaluation
    if args.eval_ppl:
        seqlen = 2048 # hard-coding
        args.limit = -1 # whole samples
        # NOTE: some keywords overlap, the first match is used
        for supported_model in SUPPORTED_MODELS:
            if supported_model in args.model:
                cache_testloader = (
                    f'calibset/wikitext_test_{seqlen}_{args.seed}_{supported_model}.cache'
                )
                logging.info(f"target calibration dataset: {cache_testloader}")
                break
        else:
            logging.warning(
                f"unrecognized model {args.model}. Loading generic calibration dataset"
            )
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}.cache'

        # =======
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader, weights_only=False)
            logging.info(f"load calibration from {cache_testloader}")
        else:
            logging.info(f"create calibration dataset for {args.model}")
            from utils.calib import get_wikitext2_test
            testloader = get_wikitext2_test(
                seed=args.seed,
                seqlen=seqlen,
                model=args.model,
                use_fast_tok=use_fast_tok,
            )
            if not os.path.exists('calibset'):
                os.mkdir('calibset')
            torch.save(testloader, cache_testloader)
            logging.info(f"calibration dataset created at {cache_testloader}")
        # =======

        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []
        with torch.no_grad():
            pbar = tqdm(range(nsamples))
            for i in pbar:
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
                outputs = None
                if "opt" in args.model.lower():
                    outputs = model.model.decoder(batch)
                elif any(
                        k in args.model.lower()
                        for k in [
                            "granite-3.0",
                            "granite-3.1",
                            "granite-3.2",
                            "granite-3.3",
                            "llama-3.1",
                            "llama-2",
                            "llama",
                            "mixtral",
                            "mistral",
                            "qwen3",
                            "qwen2",
                            "bamba",
                            "zamba2",
                            "hymba",
                            "gpt-oss",
                        ]
                    ):
                    outputs = model.model(batch)
                elif (
                    "mamba2" in args.model.lower()
                    or "falcon" in args.model.lower()
                    or "codestral" in args.model.lower()
                    or "nemotron" in args.model.lower()
                ):
                    outputs = model.backbone(batch)
                elif "qwen" in args.model.lower():  # old Qwen family
                    outputs = model.transformer(batch)
                else:
                    raise NotImplementedError(f"Model {args.model} is not supported")
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)  # (1, 2048, 49155) = (bs, seq_len, vocab)
                shift_logits = logits[:, :-1, :]

                if "granite" in args.model:
                    shift_logits /= model.config.logits_scaling

                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)  # (1, 2047) = (bs, seq_len - 1)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
                pbar.set_description(f'loss: {loss.item():.4f}')

                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        logging.info("wikitext ppl : %s", ppl.item())
        model.config.use_cache = use_cache
        results = {'wiki_ppl': ppl.item()}
    else: # lm-eval
        lm = lm_eval.models.huggingface.HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                backend='causal',
                trust_remote_code=True,
            )

        with torch.no_grad():
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
            )
        results = results['results']
    logging.info(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model and Datsets
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--tasks', type=str2list, default=[])
    parser.add_argument('--num_fewshot', type=str2int, default='none')
    parser.add_argument('--eval_ppl', type=str2bool, default=False)
    # Bit-configuration (Linear)
    parser.add_argument('--w_elem_format_linear', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_linear', type=str, default='fp4_e2m1')
    parser.add_argument('--scale_bits_linear', type=int, default=8)
    parser.add_argument('--block_size_linear', type=int, default=32)
    # Bit-configuration (MatMul)
    parser.add_argument('--A_elem_format_matmul', type=str, default='fp6_e3m2')
    parser.add_argument('--B_elem_format_matmul', type=str, default='fp4_e2m1')
    parser.add_argument('--scale_bits_matmul', type=int, default=8)
    parser.add_argument('--block_size_matmul', type=int, default=32)
    # Bit-configuration (LayerNorm)
    parser.add_argument('--w_elem_format_ln', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_ln', type=str, default='fp6_e3m2')
    parser.add_argument('--scale_bits_ln', type=int, default=8)
    parser.add_argument('--block_size_ln', type=int, default=32)
    # Bit-configuration (LM-Head)
    parser.add_argument('--w_elem_format_head', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_head', type=str, default='fp6_e3m2')
    parser.add_argument('--scale_bits_head', type=int, default=8)
    parser.add_argument('--block_size_head', type=int, default=32)
    # Others
    parser.add_argument('--auto_dtype', type=str2bool, default=True)
    parser.add_argument('--custom_cuda', type=str2bool, default=False)
    parser.add_argument('--a_scale_mode', type=int, default=0)
    parser.add_argument('--w_scale_mode', type=int, default=0)
    parser.add_argument('--A_scale_mode', type=int, default=0)
    parser.add_argument('--B_scale_mode', type=int, default=0)
    parser.add_argument('--per_tensor', type=str2bool, default=False)
    parser.add_argument('--pertensor_wscale', action='store_true')
    parser.add_argument('--pertensor_ascale', action='store_true')
    parser.add_argument('--e8m0_scale', type=float, default=-1.0)
    # Weight Scaling
    parser.add_argument('--quarot', type=str2bool, default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard')
    parser.add_argument('--rotate_kv', type=str2bool, default=True)
    parser.add_argument('--kv_quant_only', type=str2bool, default=False)
    parser.add_argument('--kv_tokenwise', type=str2bool, default=False)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(logfilename),
            logging.StreamHandler(),
        ]
    )

    parsed_args = parser.parse_args()
    set_seed(parsed_args.seed)
    logging.info(parsed_args)
    main(parsed_args)
