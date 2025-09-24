import torch
import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
from utils.utils import *
from utils.deepgemm_utils import *
import torch.nn.functional as F
import deep_gemm
import random
from deep_gemm import bench_kineto, ceil_div, get_col_major_tma_aligned_tensor, calc_diff
from typing import Tuple
import triton
import numpy as np
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()
def construct_bmm(b: int, m: int, k: int, n: int) -> \
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
              Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((b, m, k), device='cuda:0', dtype=torch.bfloat16)
    y = torch.randn((b, n, k), device='cuda:0', dtype=torch.bfloat16)
    out = torch.empty((b, m, n), device='cuda:0', dtype=torch.bfloat16)

    y = y.transpose(-2, -1)
    ref_out = torch.bmm(x, y)

    x_fp8 = to_float8(x)
    y_fp8 = to_float8(y)
    return x, y, x_fp8, y_fp8, out, ref_out

class DeepSeekEmbedding(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekEmbedding, self).__init__()
        self.tp = args.tensor_model_parallel_size
        seq_len = args.seq_length
        hidden_size = args.hidden_size
        # max_position_embeddings = args.max_position_embeddings
        self.vocab_size = args.vocab_size
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        self.device = device
        self.dtype = torch.bfloat16
        self.weight = torch.randint(
            low=0,
            high=1,  # 设置权重范围内的随机值，假设权重值在 0 到 1之间
            size=(self.vocab_size, hidden_size),
            device=device,
            dtype=torch.bfloat16 
        )
    
    @cuda_timing_decorator
    def _apply(self, input):
        if self.tp > 1:
            # Build the mask.
            input_mask = (input < 0) | (input >= math.ceil(self.vocab_size / self.tp))
            # Mask the input.
            masked_input = input.clone() - 0
            masked_input[input_mask] = 0
        else:
            masked_input = input
        
        embeddings = F.embedding(masked_input,self.weight)
        return embeddings
    def forward(self,input):
        input = input.to(self.device) 
        result, emb_time = self._apply(input)

        result = result.to(self.dtype)

        return result, emb_time

class DeepSeekRMSNorm(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekRMSNorm, self).__init__()
        self.hidden_size = args.hidden_size
        
        self.tp = args.tensor_model_parallel_size
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.norm_weight = torch.rand(self.hidden_size, device=device).to(self.dtype)
    
    @cuda_timing_decorator
    def _apply(self, hidden_states):
        output_norm = F.rms_norm(hidden_states, (self.hidden_size,), self.norm_weight, 1e-6)
        return output_norm

    def forward(self, hidden_states):
        norm_output, norm_time = self._apply(hidden_states)

        return norm_output, norm_time



class DeepSeekAtten(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekAtten, self).__init__()
        self.tp = args.tensor_model_parallel_size
        self.hidden_size = args.hidden_size
        self.num_heads = args.head_num
        self.batch_size = args.micro_batch
        self.d_kv_c =  args.d_kv_c
        self.d_q_c = args.d_q_c
        self.d_r = args.d_r
        self.d_q = args.d_q
        self.d_kv = args.d_kv

    def _qkv_compression(self):
        #不分TP
        m = self.batch_size  
        k = self.hidden_size #7168
        n = self.d_kv_c + self.d_q_c + self.d_r  #2112
        print(f'm={m}, k={k}, n={n}')
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        def test_func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t

    def _q_uncrompression(self):
        m = self.batch_size  
        k = self.d_q_c #1536
        n = self.num_heads *(self.d_q + self.d_r) // self.tp
        print(f'm={m}, k={k}, n={n}')
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        def test_func():
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t
    def _q_bmm(self):
        m = self.batch_size  
        b = self.num_heads //self.tp
        k = self.d_q
        n = self.d_kv_c #512
        print(f'm={m}, b={b}, k={k}, n={n}')
        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
        def test_func():
                        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
                        # with nvtx.range("matmul"):
                        out = torch.bmm(x_bf16, y_bf16)
        
        try:
            t = bench_kineto(test_func, "gemm_bf16", suppress_kineto_output=True)
        except:
            try:
                t = bench_kineto(test_func, "gemm_relu_bf16", suppress_kineto_output=True)
            except:
                # H20 use a strange kernel named "nvjet_tst_xxx_v_bz_TNN" to perform
                t = bench_kineto(test_func, "nvjet_tst", suppress_kineto_output=True)
        return t

    def _attention(self):
        device = torch.device("cuda:0")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(device)
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        random.seed(0)

        b = self.batch_size
        s_q = 1 #MTP的个数
        h_q = self.num_heads //self.tp
        d = self.d_kv_c + self.d_r
        q = torch.randn(b, s_q, h_q, d)
        mean_sk = 5000
        h_kv = 1
        #From DeepSeek-simulator
        cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
        for i in range(b):
            cache_seqlens[i] = max(
                random.normalvariate(mean_sk, mean_sk / 2), s_q)
        # total_seqlens = cache_seqlens.sum().item()
        # mean_seqlens = cache_seqlens.float().mean().int().item()
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        block_size = 64

        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32
            ).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
        for i in range(b):
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
                float("nan")
            )
        # blocked_v = blocked_k[..., :self.d_kv_c]
        
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, s_q * h_q // h_kv, h_kv
        )
        
        def flash_mla():
              return flash_mla_with_kvcache(
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                self.d_kv_c,
                tile_scheduler_metadata,
                num_splits,
                causal=True,
                )
        t = triton.testing.do_bench(flash_mla)
        return t
    def _O_bmm(self):
        m = self.batch_size  
        b = self.num_heads //self.tp
        k = self.d_kv_c #512
        n = self.d_kv #128
        print(f'm={m}, b={b}, k={k}, n={n}')
        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
        def test_func():
                        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
                        # with nvtx.range("matmul"):
                        out = torch.bmm(x_bf16, y_bf16)
        
        try:
            t = bench_kineto(test_func, "gemm_bf16", suppress_kineto_output=True)
        except:
            try:
                t = bench_kineto(test_func, "gemm_relu_bf16", suppress_kineto_output=True)
            except:
                # H20 uses a strange kernel named "nvjet_tst_xxx_v_bz_TNN"
                t = bench_kineto(test_func, "nvjet_tst", suppress_kineto_output=True)
        return t

    def _O_proj(self):
        m = self.batch_size
        k = self.num_heads * self.d_kv //self.tp
        n = self.hidden_size
        print(f'm={m}, k={k}, n={n}')
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        def test_func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t
         
    def forward(self):
        qkv_compression_time = self._qkv_compression()
        print("matrix 1 is:" , qkv_compression_time*1e6)
        q_uncrompression_time = self._q_uncrompression()
        print("matrix 2 is:" , q_uncrompression_time*1e6)
        q_bemm = self._q_bmm()
        print("matrix 3 is:" , q_bemm*1e6)
        atten_qkv = qkv_compression_time + q_uncrompression_time + q_bemm
        atten_core = self._attention()
        linear_bmm = self._O_bmm()
        print("matrix 9 is:" , linear_bmm*1e6)
        linear_proj = self._O_proj()
        print("matrix 4 is:" , linear_proj*1e6)
        atten_linear = linear_bmm + linear_proj
        
        return atten_qkv, atten_core, atten_linear


        
class DeepSeekMLP(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekMLP, self).__init__()
        # self.tp = 1
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = args.expert_dim
        
    
    def _up_gate(self):
        m = self.batch_size  
        k = self.hidden_size #7168
        n = self.expert_dim * 2   #不开TP
        print(f'm={m}, k={k}, n={n}')
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        def test_func():
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t

    def _down(self):
        m = self.batch_size  
        k = self.expert_dim  #不开TP
        n = self.hidden_size
        print(f'm={m}, k={k}, n={n}')
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        def test_func():
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t

    def forward(self):
        up_gate = self._up_gate()
        print("MLP:UP: ", up_gate*1e6)
        down = self._down()
        print("MLP:DOWN: ", down*1e6)
        return up_gate , down

class DeepSeekMOE(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekMOE, self).__init__()
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = args.expert_dim
        self.num_experts = args.num_experts
        self.ep = args.expert_model_parallel_size
        self.tp = args.tensor_model_parallel_size
        self.topk = args.moe_router_topk
    
    def _up_gate(self):
        num_groups = self.num_experts //self.ep  #一个GPU上有几个Expert
        dp = self.ep//self.tp
        m_per_group = dp * self.batch_size * self.topk //self.num_experts #一个expert上有几个token
        m_per_group = 2**math.ceil(math.log2(m_per_group))
        n = self.expert_dim * 2
        k = self.hidden_size

        masked_m_candidates = list(filter(
                    lambda candidate: candidate <= m_per_group,
                    (4, 8, 16, 32, 64, 128, 192, 256, 320, 384)
                ))

                # Correctness testing
        # for i in range(10):
        #     x_fp8, y_fp8, out, ref_out = construct_grouped(
        #                 num_groups, m_per_group, k, n, is_masked=True
        #             )
        #     masked_m = torch.empty(
        #                 (num_groups,), device='cuda', dtype=torch.int)
        #     for j in range(num_groups):
        #         masked_m[j] = random.choice(masked_m_candidates)
        #         expected_m = min(
        #                 int(masked_m.float().mean()) + 1, m_per_group)
        #         deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        #                 x_fp8, y_fp8, out, masked_m, expected_m
        #             )

        #     for j in range(num_groups):
        #         diff = calc_diff(
        #                     out[j, :masked_m[j].item()],
        #                     ref_out[j, :masked_m[j].item()]
        #                 )
        #         assert diff < 0.001, (
        #                     f'{m_per_group=}, {k=}, {n=}, {j=}, '
        #                     f'masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
        #                 )

        def test_func():
                    x_fp8, y_fp8, out, ref_out = construct_grouped(
                        num_groups, m_per_group, k, n, is_masked=True
                    )
                    masked_m = torch.ones(
                        (num_groups,), device='cuda:0', dtype=torch.int) * m_per_group
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                        x_fp8, y_fp8, out, masked_m, m_per_group
                    )
        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        
        return t
    
    def _down(self):
        num_groups = self.num_experts //self.ep  #一个GPU上有几个Expert
        dp = self.ep//self.tp
        m_per_group = dp * self.batch_size * self.topk //self.num_experts #一个expert上有几个token
        m_per_group = 2**math.ceil(math.log2(m_per_group))
        print("m_per_group:", m_per_group)
        n = self.hidden_size
        k = self.expert_dim
        masked_m_candidates = list(filter(
                    lambda candidate: candidate <= m_per_group,
                    (4, 8, 16, 32, 64, 128, 192, 256, 320, 384)
                ))

        def test_func():
                    x_fp8, y_fp8, out, ref_out = construct_grouped(
                        num_groups, m_per_group, k, n, is_masked=True
                    )
                    masked_m = torch.ones(
                        (num_groups,), device='cuda:0', dtype=torch.int) * m_per_group
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                        x_fp8, y_fp8, out, masked_m, m_per_group
                    )
        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        
        return t

    def forward(self):
        up_gate = self._up_gate()
        down = self._down()

        return up_gate, down


class DeepSeekModel(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekModel, self).__init__()
        self.time_list = {}
        self.args = args
        self.Embedding = DeepSeekEmbedding(self.args)
        # self.transformer = DeepSeekTransformer(self.args)
        self.RMSNorm = DeepSeekRMSNorm(self.args)
        self.Attention = DeepSeekAtten(self.args)
        self.MLP = DeepSeekMLP(self.args)
        self.MOE = DeepSeekMOE(self.args)


    def forward(self):
        # Emb_output, Emb_time = self.Embedding(input)
        # print("Embedding time: ", Emb_time)
        # self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})
        for i in range(self.args.num_layers):
            # norm_output, norm_time1 = self.RMSNorm(Emb_output)
            # self.time_list.setdefault("RMSNorm1", []).append(
            #         {"time_gpu": norm_time1}
            #     )
            atten_qkv, atten_core, atten_linear = self.Attention()
            self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv* 1e6}
                    )
            self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core* 1e3}
                    )
            self.time_list.setdefault("atten_linear", []).append(
                        {"time_gpu": atten_linear* 1e6}
                    )
            # print("norm_time time: ", norm_time1)
            print(f'qkv_core time: , {atten_qkv * 1e6}')
            print(f'qkv_attention time: , {atten_core* 1e3}')
            print(f'atten_linear time: , {atten_linear * 1e6}')
            # self.time_list.setdefault("RMSNorm2", []).append(
            #         {"time_gpu": norm_time2}
            #     )
            # print("norm_time time: ", norm_time2)
            mlp_up, mlp_down = self.MLP()
            self.time_list.setdefault("mlp_up", []).append(
                {"time_gpu": mlp_up* 1e6}
                )
            self.time_list.setdefault("mlp_down", []).append(
                {"time_gpu": mlp_down* 1e6}
            )
            print(f'mlp_time time: , {mlp_up * 1e6}')
            print(f'mlp_down time: , {mlp_down * 1e6}')
            if i >= self.args.dense_layer:
                up_gate_moe, down_moe = self.MOE()
                self.time_list.setdefault("moe_up_gate", []).append(
                        {"time_gpu": up_gate_moe* 1e6}
                    )
                self.time_list.setdefault("moe_down", []).append(
                        {"time_gpu": down_moe* 1e6}
                    )
                print(f'up_gate_moe time: , {up_gate_moe * 1e6}')  
                print(f'down_moe time: , {down_moe * 1e6}')  
        print(self.time_list)
        result_dir = "./results/aiob_outputs"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        filename = f"{self.args.model_name}_time_list_stats.txt"
        file_name = os.path.join(result_dir, filename)
        calculate_stats(self.time_list, file_name)
        filepath = write_time(self.time_list, self.args)
        process_all_keys(filepath)
        return filepath


if __name__ == "__main__":
    args = MockedDeepSeek.DeepSeekParams()
    x = torch.randint(0, args.vocab_size, (2, 1))
    model = DeepSeekModel(args)
    filepath = model(x)
    print(filepath)