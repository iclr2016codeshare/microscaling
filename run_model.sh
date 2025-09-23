#!/bin/bash

# ARGUMENTS TO THIS SCRIPT:
# $1 [required] path/to/model/and/tokenizer
# $2 [optional] GPU id(s) (runs on cuda:0 if not provided)
# $3 [optional] TAG on output log filename
echo ""
echo "========================================================"
echo "ATTENTION: call to this script was changed to:"
echo "run_model.sh <path/to/model> <gpu_ids> <tag[optional]>"
echo "========================================================"
echo ""

# ========================================================
# ARGUMENTS TO MANUALLY SET TO RUN THIS SCRIPT
MODEL_NAME=llama-3.1-8b
pertensor_wscale=false
pertensor_ascale=false
# ========================================================

# Supported MODEL_NAME in this script and microscaling/main.py:
# LLM:
#   llama-2-7b
#   llama-3.1-8b
#   llama-3.1-8b-instruct
#   granite-3.2-8b-instruct
#   granite-3.3-8b-base
#   granite-3.3-8b-instruct
#   qwen2.5-7b-instruct
#   qwen3-8b
#   codestral-7b
# SSM & hybrids:
#   bamba-9b
#   bamba-9b-v2
#   mamba2
#   zamba2-7b
#   hymba-1.5b-base
#   hymba-1.5b-instruct
#   falcon3-mamba-7b-base
#   falcon3-mamba-7b-instruct
#   nemotron-nano-9b-v2
# MOE:
#   mixtral-8x7b-instruct
#   llama-4-scout-17b-16e-instruct
#   gpt-oss-20b
#   gpt-oss-120b


if [ ! "$1" ]; then
    echo "Error: path to model folder of ${MODEL_NAME} must be provided."
    exit 1
fi

ROOTDIR=${ROOTDIR:-$HOME/microscaling}
if [[ $MODEL_NAME == "llama-2-7b" ]]; then
    LOGDIR="llama2"
elif [[ $MODEL_NAME == "llama-3.1-8b" ]]; then
    LOGDIR="llama3"
elif [[ $MODEL_NAME == "llama-3.1-8b-instruct" ]]; then
    LOGDIR="llama3"
elif [[ $MODEL_NAME == "llama-4-scout-17b-16e-instruct" ]]; then
    LOGDIR="llama4"
elif [[ $MODEL_NAME == "granite-3.2-8b-instruct" ]]; then
    LOGDIR="granite3"
elif [[ $MODEL_NAME == "granite-3.3-8b-base" ]]; then
    LOGDIR="granite3"
elif [[ $MODEL_NAME == "granite-3.3-8b-instruct" ]]; then
    LOGDIR="granite3"
elif [[ $MODEL_NAME == "qwen2.5-7b-instruct" ]]; then
    LOGDIR="qwen"
elif [[ $MODEL_NAME == "qwen3-8b" ]]; then
    LOGDIR="qwen"
elif [[ $MODEL_NAME == "codestral-7b" ]]; then
    LOGDIR="codestral"
elif [[ $MODEL_NAME == "bamba-9b" ]]; then
    LOGDIR="bamba"
elif [[ $MODEL_NAME == "bamba-9b-v2" ]]; then
    LOGDIR="bamba"
elif [[ $MODEL_NAME == "mamba2" ]]; then
    LOGDIR="mamba2"
elif [[ $MODEL_NAME == "zamba2" ]]; then
    LOGDIR="zamba"
elif [[ $MODEL_NAME == "hymba-1.5b-base" ]]; then
    LOGDIR="hymba"
elif [[ $MODEL_NAME == "hymba-1.5b-instruct" ]]; then
    LOGDIR="hymba"
elif [[ $MODEL_NAME == "falcon3-mamba-7b-base" ]]; then
    LOGDIR="falcon3"
elif [[ $MODEL_NAME == "falcon3-mamba-7b-instruct" ]]; then
    LOGDIR="falcon3"
elif [[ $MODEL_NAME == "mixtral-8x7b-instruct" ]]; then
    LOGDIR="mixtral"
elif [[ $MODEL_NAME == "gpt-oss-20b" ]]; then
    LOGDIR="gpt-oss"
elif [[ $MODEL_NAME == "gpt-oss-120b" ]]; then
    LOGDIR="gpt-oss"
elif [[ $MODEL_NAME == "nemotron-nano-9b-v2" ]]; then
    LOGDIR="nemotron"
else
    echo "Model ${MODEL_NAME} not recognized"
    exit 1
fi
if [ "x$2" == "x" ]; then
    GPUIDS=0
else
    GPUIDS=$2
fi
echo "------------------------------------------------"
echo "Running ${MODEL_NAME} on GPU ${GPUIDS}"
echo "------------------------------------------------"

LOGDIR_FULL="${ROOTDIR}/${LOGDIR}"
if [ ! -d $LOGDIR_FULL ]; then
    mkdir $LOGDIR_FULL
fi
echo "Output folder: ${LOGDIR_FULL}"

model=none  # path to checkpoint must be passed as argument #2
seed=71731
tasks=none  # piqa,hellaswag,winogrande,gsm8k,mmlu
num_fewshot=none  # usually 5 if enabled
eval_ppl=true

w_elem_format_linear=none
a_elem_format_linear=none
scale_bits_linear=8
block_size_linear=32

A_elem_format_matmul=none
B_elem_format_matmul=none
scale_bits_matmul=8
block_size_matmul=32

w_elem_format_ln=none
a_elem_format_ln=none
scale_bits_ln=8
block_size_ln=32

w_elem_format_head=none  # fp4_e2m1 fp4_e2m1_asym fp8_e5m2  none
a_elem_format_head=none  # fp4_e2m1 fp4_e2m1_asym fp8_e5m2  none
scale_bits_head=8
block_size_head=32

auto_dtype=true
custom_cuda=true
a_scale_mode=0
w_scale_mode=0
A_scale_mode=0
B_scale_mode=0
per_tensor=false

quarot=false  # true
rotate_mode=hadamard
rotate_kv=false   # true
kv_quant_only=false
kv_tokenwise=false

# ======
# customize script call and log string
if [ $w_elem_format_head = "none" ] && [ $a_elem_format_head = "none" ] ; then
    HEADQ_STR="_head-noQ"
else
    HEADQ_STR="_headW-${w_elem_format_head}_headA-${w_elem_format_head}"
fi
if [ $pertensor_wscale = "true" ]; then
    PERT_WSCALE_ARG="--pertensor_wscale"
    if [ $pertensor_ascale = "true" ]; then
        PERT_ASCALE_ARG="--pertensor_ascale"
        PERT_LOG_STR="_TensorScaleWA"
    else
        PERT_ASCALE_ARG=""
        PERT_LOG_STR="_TensorWeightScale"
    fi
else
    PERT_WSCALE_ARG=""
    if [ $pertensor_ascale = "true" ]; then
        PERT_ASCALE_ARG="--pertensor_ascale"
        PERT_LOG_STR="_TensorActivScale"
    else
        PERT_ASCALE_ARG=""
        PERT_LOG_STR=""
    fi
fi
if [ "$3" ]; then
    TAG="_$3"
fi
# ======

for model in $1; do
    for scale_bits in 8; do
        scale_bits_linear=$scale_bits
        scale_bits_matmul=$scale_bits
        for per_tensor in false; do
            for block_size in 8; do
                block_size_linear=$block_size
                block_size_matmul=$block_size
                block_size_head=$block_size
                for format in fp4_e2m1; do  # fp4_e2m1_asym
                    w_elem_format_linear=$format
                    a_elem_format_linear=$format
                    for mm_format in none; do
                        A_elem_format_matmul=$mm_format
                        B_elem_format_matmul=$mm_format
                        #   0 = E8M0
                        # 143 = FP8E4M3
                        # 152 = FP8E5M2
                        #  53 = FP8E5M3
                        #  16 = BF16
                        for scale_mode in 143 53 16; do
                            w_scale_mode=$scale_mode
                            a_scale_mode=$scale_mode
                            for mm_scale_mode in 0; do
                                A_scale_mode=$mm_scale_mode
                                B_scale_mode=$mm_scale_mode
                                for e8m0_scale in -1; do

set -x # echo on
CUDA_VISIBLE_DEVICES=$GPUIDS python main.py \
    --model=$model \
    --seed=$seed \
    --tasks=$tasks \
    --num_fewshot=$num_fewshot \
    --eval_ppl=$eval_ppl \
    --w_elem_format_linear=$w_elem_format_linear \
    --a_elem_format_linear=$a_elem_format_linear \
    --scale_bits_linear=$scale_bits_linear \
    --block_size_linear=$block_size_linear \
    --A_elem_format_matmul=$A_elem_format_matmul \
    --B_elem_format_matmul=$B_elem_format_matmul \
    --scale_bits_matmul=$scale_bits_matmul \
    --block_size_matmul=$block_size_matmul \
    --w_elem_format_ln=$w_elem_format_ln \
    --a_elem_format_ln=$a_elem_format_ln \
    --scale_bits_ln=$scale_bits_ln \
    --block_size_ln=$block_size_ln \
    --w_elem_format_head=$w_elem_format_head \
    --a_elem_format_head=$a_elem_format_head \
    --scale_bits_head=$scale_bits_head \
    --block_size_head=$block_size_head \
    --auto_dtype=$auto_dtype \
    --custom_cuda=$custom_cuda \
    --a_scale_mode=$a_scale_mode \
    --w_scale_mode=$w_scale_mode \
    --A_scale_mode=$A_scale_mode \
    --B_scale_mode=$B_scale_mode \
    --per_tensor=$per_tensor \
    --quarot=$quarot \
    --rotate_mode=$rotate_mode \
    --rotate_kv=$rotate_kv \
    --kv_quant_only=$kv_quant_only \
    --kv_tokenwise=$kv_tokenwise \
    ${PERT_WSCALE_ARG} \
    ${PERT_ASCALE_ARG} \
2>&1 | tee ${LOGDIR_FULL}/${MODEL_NAME}_\
WA-${format}_S-${scale_mode}_BMM-${mm_format}_Smm-${mm_scale_mode}\
${HEADQ_STR}_mxbs${block_size}${PERT_LOG_STR}_${tasks}${TAG}.log
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
set +x # echo off
