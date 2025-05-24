"""Provides DeepSeek implementation for Aiob compute time calculations.

Based on https://github.com/deepseek-ai/DeepSeek-V3/tree/f6e34dd26772dd4a216be94a8899276c5dca9e43
and https://github.com/deepseek-ai/DeepGEMM/tree/391755ada0ffefa9a6a52b6f14dcaf22d1a463e0

Note: DeepGEMM sadly only works with SM90, SM100 i.e. Hopper and Blackwell

Todo:
- add FlashMLA from https://github.com/deepseek-ai/FlashMLA/ for attentions
    as of https://github.com/deepseek-ai/FlashMLA/commit/41b611f7d7561790a2f5040ff89212e08c7b0011 
    FlashMLA's kernel supports backwards :)
- add DeepEP from https://github.com/deepseek-ai/DeepEP/

@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}


File: AiobDeepSeek.py
License: Apache 2.0

"""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.utils import cuda_timing_decorator

# needs deep_gemm from https://github.com/deepseek-ai/DeepGEMM
try:
    import deep_gemm
    from deep_gemm import ceil_div
except ImportError as e:
    print("""
        Needs deep_gemm from https://github.com/deepseek-ai/DeepGEMM
        to install,
           git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
           cd DeepGEMM
           ./install.sh
    """)
    raise e


# from deepgemm/tests
def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2),
    )


@dataclass
class Args:
    hidden_size: int
    ffn_hidden_size: int
    micro_batch: int
    expert_model_parallel_size: int
    num_experts: int
    moe_router_topk: int
    seq_length: int
    n_shared_expert: int


class DeepSeekExpert(torch.nn.Module):
    """DeepSeekExpert"""

    def __init__(self, args: Args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.dev = torch.cuda.current_device()
        self.w1 = torch.randn(
            args.ffn_hidden_size,
            args.hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)
        self.w1_fp8 = per_block_cast_to_fp8(self.w1)

        self.w2 = torch.randn(
            args.hidden_size,
            args.ffn_hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)

        self.w2_fp8 = per_block_cast_to_fp8(self.w2)

        self.w3 = torch.randn(
            args.ffn_hidden_size,
            args.hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)

        self.w3_fp8 = per_block_cast_to_fp8(self.w3)

        self.o0 = torch.empty(
            (args.seq_length, self.ffn_hidden_size),
            device=self.dev,
            dtype=torch.bfloat16,
        )
        self.o1 = torch.empty(
            (args.seq_length, self.ffn_hidden_size),
            device=self.dev,
            dtype=torch.bfloat16,
        )
        self.out = torch.empty(
            (args.seq_length, self.hidden_size), device=self.dev, dtype=torch.bfloat16,
        )

    @cuda_timing_decorator
    def _apply_linear1(self, x: torch.Tensor, x_scale: torch.Tensor):
        deep_gemm.gemm_fp8_fp8_bf16_nt((x, x_scale), self.w1_fp8, self.o0)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x, x_scale), self.w3_fp8, self.o1)
        return (self.o0, self.o1)

    @cuda_timing_decorator
    def _apply_linear2(self, x: torch.Tensor):
        self.out = torch.matmul(x, self.w2.T)
        # or one can scale it down to fp8 and apply deepgemm
        # x_fp8 = per_token_cast_to_fp8(x)
        # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, self.w2_fp8, self.out)
        return self.out

    @cuda_timing_decorator
    def _apply_activation(self, x: torch.Tensor):
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        # based on https://github.com/deepseek-ai/DeepSeek-V3/blob/f6e34dd26772dd4a216be94a8899276c5dca9e43/inference/model.py#L630
        # i.e. out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x, x_scale = per_token_cast_to_fp8(x.view(-1, self.hidden_size))
        l1_out, l1_time = self._apply_linear1(x, x_scale)
        act_out, act_time = self._apply_activation(l1_out[0])
        l2_out, l2_time = self._apply_linear2(act_out * l1_out[0])
        return l2_out, l1_time, act_time, l2_time


class DeepSeekMoE(torch.nn.Module):
    """DeepSeekMoE"""

    def __init__(self, args: Args):
        super().__init__()
        self.n_local_experts = args.n_shared_expert + args.num_experts // args.expert_model_parallel_size
        self.hidden_size = args.hidden_size
        self.dev = torch.cuda.current_device()
        in_dim = self.hidden_size * args.micro_batch
        self.in_dim = in_dim
        self.inter_dim = args.ffn_hidden_size * self.n_local_experts
        # yoinked from AiobMegatron.py->MoELayer
        ep = args.expert_model_parallel_size
        num_experts = args.num_experts
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        topk = args.moe_router_topk
        self.topk = topk
        hidden_size = args.hidden_size
        self.dispatched_input = torch.rand(
            int(seq_len * micro_batch * topk * ep / num_experts * self.n_local_experts),
            hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)
        self.dispatched_input_fp8 = per_token_cast_to_fp8(self.dispatched_input)

        self.tokens_per_expert = torch.full(
            (self.n_local_experts,),
            int(seq_len * micro_batch * topk * ep / num_experts),
        )

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.n_local_experts):
            self.local_experts.append(DeepSeekExpert(args))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int,int,int]:
        mlp_l1_all, mlp_act_all, mlp_l2_all = 0, 0, 0
        o = torch.zeros(
            self.n_local_experts, x.shape[0], self.hidden_size, device=self.dev,
        ).to(torch.bfloat16)
        for i, expert in enumerate(self.local_experts):
            out, mlp_l1, mlp_act, mlp_l2 = expert(x.view(-1, self.in_dim))
            mlp_l1_all += mlp_l1
            mlp_act_all += mlp_act
            mlp_l2_all += mlp_l2
            o[i] = out
        return o, mlp_l1_all, mlp_act_all, mlp_l2_all


# some util tests to measure timings
def test_deepseek_expert() -> torch.Tensor:
    x = torch.randn(2048, 1, 1024, dtype=torch.bfloat16)
    args = Args(
        expert_model_parallel_size=1,
        ffn_hidden_size=4096,
        hidden_size=1024,
        num_experts=8,
        micro_batch=1,
        moe_router_topk=4,
        n_shared_expert=2,
        seq_length=2048,
    )
    model = DeepSeekExpert(args)
    y = model(x)
    print(y)


def test_deepseek_mlp():
    x = torch.randn(2048, 1, 1024, dtype=torch.bfloat16)
    args = Args(
        expert_model_parallel_size=1,
        ffn_hidden_size=4096,
        hidden_size=1024,
        num_experts=8,
        micro_batch=1,
        moe_router_topk=4,
        n_shared_expert=2,
        seq_length=2048,
    )
    model = DeepSeekMoE(args)
    y = model(x)
    return y


def test_expert(
    m: int = 2048 * 32, n: int = 1024, k: int = 2048 * 32,
) -> tuple[int, int]:
    """Test DeepSeek's expert MLP.

    i.e. self.w2(F.silu(self.w1(x)) * self.w3(x))
    with torch.matmul (BF16) and deep_gemm (FP8).
    """
    x = torch.randn(m, k).to(torch.bfloat16)
    w = torch.randn(k, n).to(torch.bfloat16)
    w = w.t()
    w2 = torch.randn(k, n).to(torch.bfloat16)
    w2 = w2.t()
    o = torch.randn(m, n).to(torch.bfloat16)
    o2 = torch.randn(m, n).to(torch.bfloat16)

    w3 = torch.randn(n, k).to(torch.bfloat16)
    w3_fp8 = per_block_cast_to_fp8(w3.t())

    out = torch.randn(m, k).to(torch.bfloat16)
    w_fp8 = per_block_cast_to_fp8(w)
    w2_fp8 = per_block_cast_to_fp8(w2)
    x_fp8 = per_token_cast_to_fp8(x)

    _, t0 = do_deepgemm(x_fp8, w_fp8, w2_fp8, w3_fp8, o, o2, out)
    _, t1 = do_matmul(x, w, w3, o, out)
    return t0, t1


@cuda_timing_decorator
def do_deepgemm(
    x: tuple[torch.Tensor, torch.Tensor],
    w: tuple[torch.Tensor, torch.Tensor],
    w2: tuple[torch.Tensor, torch.Tensor],
    w3: torch.Tensor,
    o: torch.Tensor,
    o2: torch.Tensor,
    out: torch.Tensor,
) -> tuple[int,int]:
    deep_gemm.gemm_fp8_fp8_bf16_nt(x, w, o)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x, w2, o2)
    o2 = F.silu(o) + o2
    o2_fp8 = per_token_cast_to_fp8(o2)
    deep_gemm.gemm_fp8_fp8_bf16_nt(o2_fp8, w3, out)
    return out


@cuda_timing_decorator
def do_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    w3: torch.Tensor,
    o: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.matmul(x, w.t(), out=o)
    return torch.matmul(F.silu(o), w3, out=out)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    # run once to load/cache kernels
    test_deepseek_mlp()
    test_expert()

    # for kineto traces
    import torch.profiler

    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        test_deepseek_mlp()
        t0, t1 = test_expert()
    print(f"matmul: deepgemm: {t0} matmul: {t1} diff: {t1 - t0}")
