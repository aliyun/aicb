"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Dict
import pandas as pd
import pickle
from enum import Enum
import argparse
import sys
import time
import os
import json
from collections import defaultdict
import math
import re

try:
    import torch
except ImportError as e:
    torch = None
    print("Failed to import 'torch'.")


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def Comp_with_aiob(workload, compute_cache):
    for item in workload.workload:
        if item.comm_type == CommType.computation:
            for key in compute_cache:
                key_temp = key.split("_")[0]
                if key_temp in item.stage:
                    item.msg_size = compute_cache[key]
                    break
    return workload


def get_comp_out(args):
    vocab_size = args.vocab_size
    batch_size = args.micro_batch
    seq_len = args.seq_length
    tp = args.tp_num
    vocab_size = args.padded_vocab_size
    if "Megatron" in args.comm_frame:
        device = torch.cuda.current_device()
        from workload_generator.mocked_model.AiobMegatron import MegatronModel

        measure_model = MegatronModel(args)
        measure_model.train()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        # total_input_1 = torch.rand(args.seq_len,
        #                                       args.batch_size,
        #                                       args.hidden_size,
        #                                       device=device).to(dtype)
        masked_input = torch.randint(
            0,
            math.ceil(vocab_size / tp),
            (batch_size, seq_len),
            device=device,
            dtype=torch.int64,
        )
        filepath = measure_model(masked_input)
    return filepath


def extract_averages(file_path):
    attention_avg_sum = 0.0
    mlp_avg_sum = 0.0
    other_avgs = {}
    grad_forward = 0.0
    grad_backward = 0.0

    section_header_re = re.compile(r"^(\w+):")
    time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")
    time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")

    with open(file_path, "r") as file:
        current_section = None

        for line in file:
            header_match = section_header_re.match(line)
            if header_match:
                current_section = header_match.group(1).strip()

            avg_match = time_gpu_avg_re.search(line)
            min_match = time_gpu_min_re.search(line)
            if current_section == "param_time":
                if min_match:
                    grad_forward = float(min_match.group(1)) * 1000 #us
                if avg_match:
                    grad_backward = float(avg_match.group(1)) * 1000
            elif avg_match and current_section:
                avg_value = float(avg_match.group(1)) * 1000
                if "atten" in current_section or current_section == "layernorm":
                    attention_avg_sum += avg_value
                elif "mlp" in current_section or current_section == "layernorm2":
                    mlp_avg_sum += avg_value
                else:
                    other_avgs[current_section] = avg_value

    # 四舍五入并转换为整数
    attention_forward = round(attention_avg_sum)
    attention_backward = attention_forward
    mlp_forward = round(mlp_avg_sum)
    mlp_backward = mlp_forward
    grad_backward = round(grad_backward)
    grad_forward = round(grad_forward)
    other_avgs_int = {k: round(v) for k, v in other_avgs.items() if k != "param_time"}

    a100_compute_cache = {
        "attention_forward": attention_forward,
        "attention_backward": attention_backward,
        "mlp_forward": mlp_forward,
        "mlp_backward": mlp_backward,
        "grad_forward": grad_forward,
        "grad_backward": grad_backward,
    }
    a100_compute_cache.update(other_avgs_int)

    return a100_compute_cache


def process_all_keys(input_file):

    with open(input_file, "r") as file:
        first_line_str = file.readline().strip()
        remaining_content = file.read().strip()
    # 尝试修复和构建合法的 JSON 字符串
    corrected_content = remaining_content.replace("}{", "},{").replace("]}{", "]},{")

    # 构建 JSON 数组
    json_str = f"[{corrected_content}]"

    try:
        data = json.loads(json_str)

        processed_results = defaultdict(lambda: defaultdict(list))
        for entry in data:
            for key, measurements in entry.items():
                for measure in measurements:
                    measure_key, measure_value = next(iter(measure.items()))
                    if "time_gpu" in measure_key:
                        processed_results[key]["time_gpu"].append(measure["time_gpu"])
                    else:
                        processed_results[key][measure_key] = measure_value

        for key, values in processed_results.items():
            if "time_gpu" in values:
                gpu_times = values["time_gpu"]
                min_time_gpu = min(gpu_times)
                gpu_times_filtered = [t for t in gpu_times if t <= 3 * min_time_gpu]
                values["time_gpu_max"] = max(gpu_times_filtered)
                values["time_gpu_min"] = min_time_gpu
                values["time_gpu_avg"] = sum(gpu_times_filtered) / len(
                    gpu_times_filtered
                )
                del values["time_gpu"]

        with open(input_file, "w") as outfile:
            outfile.write(first_line_str + "\n")
            for key, values in processed_results.items():
                outfile.write(f"{key}:\n")
                for subkey, subvalue in values.items():
                    outfile.write(f"    {subkey}: {subvalue}\n")
        print(f"Compute-results save in:{input_file}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print("Invalid JSON content:\n", corrected_content)


def cuda_timing_decorator(func):
    def wrapper(*args, **kwargs):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event) * 1000  # 时间以毫秒为单位
        return result, elapsed_time_ms

    return wrapper


def write_op(time_list, args):
    result_dir = "./results/aiob_outputs"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tp_num}-pp{args.pp_num}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-flash_attn-{args.use_flash_attn}.txt"
    filepath = os.path.join(result_dir, filename)
    with open(filepath, "w") as file:
        file.write(f"train_iter:{args.epoch_num}\n")
        data_str = json.dumps(time_list, indent=4)

        file.write(data_str)
    return filepath


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


class CommType(str, Enum):
    """Enum class for possible comm types"""

    all_reduce = "all_reduce"
    isend = "isend"
    irecv = "irecv"
    broadcast = "broadcast"
    all_gather = "all_gather"
    reduce_scatter = "reduce_scatter"
    barrier = "barrier"
    reduce = "reduce"
    reduce_scatter_tensor = "reduce_scatter_tensor"
    all_gather_into_tensor = "all_gather_into_tensor"
    computation = "computation"
    epoch_end = "epoch_end"
    all_to_all = "all_to_all"

    @classmethod
    def get_comm_type(cls, value):
        for comm_type in cls:
            if comm_type.value == value:
                return comm_type
        raise ValueError("Invailid communication type")


class CommGroup(str, Enum):
    """Enum class for possible comm groups"""

    dp_group = "dp_group"
    pp_group = "pp_group"
    tp_group = "tp_group"
    ep_group = "ep_group"
    ep_dp_group = "ep_dp_group"
    ep_tp_group = "ep_tp_group"
    embedding_group = "embedding_group"
    all = "all_nodes"


class WorkloadWriter:
    @staticmethod
    def write_workload(workload: List[Dict], args, filename: str):
        df = pd.DataFrame.from_dict(workload)
        df = df.fillna(-1)
        df.to_csv(filename, index=False)
        filename = filename.split(".")[0] + ".pkl"
        pickle.dump((workload, args), open(filename, "wb"))

    @staticmethod
    def load_workload(filename: str) -> List[Dict]:
        filename = filename.split(".")
        filename[-1] = "pkl"
        filename = ".".join(filename)
        workload, args = pickle.load(open(filename, "rb"))
        return workload, args


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comm_frame",
        help="communication framework",
        choices=["Megatron", "DeepSpeed", "collective_test"],
        default="Megatron",
    )
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--tp_num", type=int, default=1,
                        help='Degree of tensor model parallelism.')
    parser.add_argument("--pp_num", type=int, default=1,
                        help='Degree of pipeline model parallelism.')
    parser.add_argument("--pp_rank", type=int, default=-1,
                        help='Rank where encoder and decoder should be split.')
    parser.add_argument("--global_batch", type=int, default=4,
                        help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'
                        )
    parser.add_argument("--epoch_num", type=int, default=1,
                        help="Number of iterations")
    parser.add_argument("--computation_enable", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=None,
        help="Transformer Feed-Forward Network hidden size. "
        "This is set to 4*hidden-size if not provided",
    )
    get_model_params(parser)
    get_ds_params(parser)
    get_megatron_params(parser)
    get_collective_test_params(parser)
    get_moe_params(parser)
    get_simAI_workload_params(parser)
    get_aiob_params(parser)
    args = parser.parse_args()

    assert (
        args.world_size % (args.tp_num * args.pp_num) == 0
    ), f"world size: {args.world_size}, tp: {args.tp_num}, pp: {args.pp_num}"
    assert (
        args.moe_enabled and args.enable_sequence_parallel
    ), f"moe must be enabled with sequence parallel"
    args.dp_num = args.world_size // (args.tp_num * args.pp_num)
    # assert args.global_batch % (args.dp_num * args.micro_batch) == 0, \
    #     f"global_batch: {args.global_batch}, dp: {args.dp_num}, micro_batch: {args.micro_batch}"
    args.num_microbatches = args.global_batch // (args.dp_num * args.micro_batch)

    if args.num_attention_heads is None:
        args.num_attention_heads = args.num_layers

    args.padded_vocab_size = get_padded_vocab_size(args)
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

        else:
            args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.gated_linear_unit = True
        args.bias_gelu_fusion = False
    return args


ARGS = None


def get_args():
    global ARGS
    if ARGS is not None:
        return ARGS
    ARGS = get_params()
    return ARGS


def get_aiob_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--aiob_enable",
        action="store_true",
        help="Enable aiob to get operation real compute time",
    )
    parser.add_argument("--comp_filepath", type=str, default=None,
                        help="Use aiob_lib to get operation real compute time",)
    parser.add_argument("--gated_linear_unit", default=False)
    parser.add_argument("--bias_gelu_fusion", action="store_true",
                        help='Enable bias and gelu fusion.')
    parser.add_argument("--openai_gelu", action="store_true",
                         help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    parser.add_argument("--onnx_safe", action="store_true",
                        help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    parser.add_argument("--squared_relu", action="store_true",
                        help='Use squared relu activation instead of default gelu')


def get_model_params(parser: argparse.ArgumentParser):
    parser.add_argument("--model_name", help="Model for training")
    parser.add_argument(
        "--hidden_size", type=int, help='Tansformer hidden size.', default=1024
    )
    parser.add_argument("--num_layers", type=int, help='Number of transformer layers.', default=24)
    parser.add_argument(
        "--seq_length", type=int, help='Maximum sequence length to process.', default=2048
    )
    parser.add_argument("--num_attention_heads", help='Number of transformer attention heads.',type=int, default=None)
    parser.add_argument("--vocab_size", type=int, help='Size of vocab before EOD or padding.', default=32000)
    parser.add_argument("--max_position_embeddings", type=int,help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.', default=4096)
    parser.add_argument("--add_bias_linear",help='Enable bias in the linear layers', action="store_true")
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use FlashAttention implementation of attention.",
    )
    parser.add_argument(
        "--swiglu",
        action="store_true",
        help="Use gated linear units and SiLU activation instead of default gelu",
    )


def get_ds_params(parser: argparse.ArgumentParser):
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--amp_enabled", action="store_true")
    parser.add_argument("--reduce_bucket_size", type=int, default=int(5e8))

    # for stage1/2 only
    parser.add_argument("--allgather_bucket_size", type=int, default=int(5e8))
    parser.add_argument("--contiguous_gradients", action="store_true")

    # for stage 3 only
    parser.add_argument("--param_persistence_threshold", type=int, default=int(1e5))
    parser.add_argument(
        "--model_persistence_threshold", type=int, default=int(sys.maxsize)
    )
    parser.add_argument("--max_live_parameters", type=int, default=int(1e9))
    parser.add_argument("--prefetch_bucket_size", type=int, default=int(1e9))


def get_megatron_params(parser: argparse.ArgumentParser):
    parser.add_argument("--enable_sequence_parallel",help='Enable sequence parallel optimization.',action="store_true")
    parser.add_argument(
        "--use-distributed-optimizer",
        action="store_true",
        help="Use distributed optimizer.",
    )
    parser.add_argument("--make_vocab_size_divisible_by", help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.',type=int, default=128)
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        default=False,
        help="If set, overlap DDP grad reduce. (Not implement yet)",
    )


def get_collective_test_params(parser: argparse.ArgumentParser):
    parser.add_argument("--begin_size", type=int, default=1048576)
    parser.add_argument("--end_size", type=int, default=1048576)
    parser.add_argument("--test_comm", type=str, default="all_reduce")
    parser.add_argument("--iter_num", type=int, default=500)
    parser.add_argument("--multi_all_reduce_enable", type=int, default=0)


def get_simAI_workload_params(parser: argparse.ArgumentParser):
    parser.add_argument("--overlap_version", action="store_true")

def get_moe_params(parser: argparse.ArgumentParser):
    parser.add_argument('--moe_enabled', action="store_true")
    parser.add_argument('--expert_parallel_size', type=int, default=1, help='Degree of expert model parallelism.')
    parser.add_argument('--num_moe_experts', type=int, default=1, help='Number of Experts in MoE (None means no MoE)')
    parser.add_argument('--moe_router_topk', type=int, default=1, help='Number of experts to route to for each token. The default is 2.')
    parser.add_argument('--moe_grouped_gemm', action='store_true',
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')
    parser.add_argument('--activation_func', type=str,help='activation_func for mlp')

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def get_padded_vocab_size(args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = args.vocab_size

    multiple = args.make_vocab_size_divisible_by * args.tp_num
    while (after % multiple) != 0:

        after += 1

    return after


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
