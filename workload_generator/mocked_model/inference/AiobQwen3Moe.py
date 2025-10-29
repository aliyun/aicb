import torch
import workload_generator.mocked_model.inference.MockedQwen3Moe as MockedQwen3Moe
from utils.utils import *
from utils.deepgemm_utils import *
import torch.nn.functional as F
import deep_gemm
import random
from deep_gemm import bench_kineto, ceil_div, get_col_major_tma_aligned_tensor, calc_diff
from typing import Optional, Union, Tuple
# import triton
import numpy as np
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
import flashinfer

# from benchmark_rmsnorm.py
def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output

class Qwen3MoeAttention(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3MoeAttention, self).__init__()
        self.args = args
        self.tp = self.args.tensor_model_parallel_size

    def _norm(self):
        m = self.args.micro_batch
        head_dim = self.args.head_dim
        x = torch.randn(m,head_dim, device='cuda', dtype=torch.bfloat16)
        weight = torch.ones(head_dim, dtype=torch.bfloat16, device='cuda')
        residual = torch.randn(m,head_dim, device='cuda', dtype=torch.bfloat16)
        def test_func():
            rmsnorm_vllm(x.clone(), weight, residual.clone(), self.args.rms_norm_eps)
        t = bench_kineto(test_func, 'vllm::fused_ad', suppress_kineto_output=True)
        return t
    def _attn_quant(self, qk_or_o):
        m = self.args.micro_batch
        if qk_or_o:
            n = self.args.hidden_size
        else:
            n = self.args.num_attention_heads * self.args.head_dim // self.tp
        input = torch.randn(m, n, device='cuda', dtype=torch.bfloat16)
        def test_func():
            per_token_group_quant_fp8(input, 128, column_major_scales=True)

        candidates = [
            'per_token_group_quant_fp8', # old version
            'per_token_group_quant_8bit', # new version
        ]
        last_err = None
        for names in candidates:
            try:
                t = bench_kineto(test_func, names, suppress_kineto_output=True)
                return t
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")

    def _attn_proj(self, qk_or_o):
        m = self.args.micro_batch
        if qk_or_o:
            n = (self.args.num_attention_heads + self.args.num_key_value_heads * 2) * self.args.head_dim // self.tp
            k = self.args.hidden_size
        else:
            n = self.args.hidden_size
            k = self.args.num_attention_heads  * self.args.head_dim // self.tp
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        def test_func():
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t
    def _qk_norm(self, q_or_k):
        m = self.args.micro_batch
        head_dim = self.args.head_dim
        if q_or_k:
            n = self.args.num_attention_heads * head_dim // self.tp
        else:
            n = self.args.num_key_value_heads * head_dim // self.tp
        x = torch.randn(m,n, device='cuda', dtype=torch.bfloat16)
        x = x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)
        weight = torch.ones(head_dim, dtype=torch.bfloat16, device='cuda')
        def test_func():
            rmsnorm_vllm(x.clone(), weight, residual=None, eps = self.args.rms_norm_eps)
        t = bench_kineto(test_func, 'vllm::rms_norm_kernel', suppress_kineto_output=True)
        return t
            
    def _rotary_emb(self):
        m = self.args.micro_batch
        head_dim = self.args.head_dim
        n_q = self.args.num_attention_heads * head_dim // self.tp
        n_k = self.args.num_key_value_heads * head_dim // self.tp

        q = torch.randn(m, n_q, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(m, n_k, device='cuda', dtype=torch.bfloat16)
        pos = torch.arange(0, m, device='cuda')
        rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=self.args.max_position_embeddings,
                base=self.args.rope_theta,
                rope_scaling=self.args.rope_scaling,
                dual_chunk_attention_config = getattr(self.args, 
                                        "dual_chunk_attention_config",
                                        None),
            )
        def test_func():
            rotary_emb(pos, q, k)

        t = bench_kineto(test_func, 'rotary_embedding_kernel', suppress_kineto_output=True)
        return t

    def _attn_core(self):
        block_size = 32
        num_blocks = 32768 
        
        seq_len = self.args.seq_length
        kv_lens = [seq_len] * self.args.micro_batch
        head_size = self.args.head_dim
        num_query_heads = self.args.num_attention_heads // self.tp
        num_kv_heads = self.args.num_key_value_heads// self.tp
        dtype = torch.bfloat16


        # based on vllm:test_flashinfer_decode_with_paged_kv
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        soft_cap = None

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_value_cache = torch.randn(num_blocks,
                                    2,
                                    block_size,
                                    num_kv_heads,
                                    head_size,
                                    dtype=dtype)
        key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
        value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0,
                                    num_blocks,
                                    (num_seqs, max_num_blocks_per_seq),
                                    dtype=torch.int32)

        kv_indptr = [0]
        kv_indices = []
        kv_last_page_lens = []
        for i in range(num_seqs):
            seq_len = kv_lens[i]
            assert seq_len > 0
            num_blocks = (seq_len + block_size - 1) // block_size
            kv_indices.extend(block_tables[i, :num_blocks])
            kv_indptr.append(kv_indptr[-1] + num_blocks)
            kv_last_page_len = seq_len % block_size
            if kv_last_page_len == 0:
                kv_last_page_len = block_size
            kv_last_page_lens.append(kv_last_page_len)

        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)
        
        # 128 MB 
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
        wrapper = flashinfer.\
            BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD",
                    use_tensor_cores=(
                        (num_query_heads//num_kv_heads) > 4)
                    )
        wrapper.plan(kv_indptr,
                    kv_indices,
                    kv_last_page_lens,
                    num_query_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    "NONE",
                    q_data_type=dtype,
                    kv_data_type=dtype,
                    logits_soft_cap=soft_cap)
        
        def test_func():
            wrapper.run(query, key_value_cache)
        t = bench_kineto(test_func, 'BatchPrefillWithPagedKVCacheKernel', suppress_kineto_output=True)
        return t

    def forward(self):
        #norm
        pre_attention_layernorm = self._norm()

        # qkv
        qkv_quant = self._attn_quant(True)
        qkv_proj = self._attn_proj(True)
        q_norm = self._qk_norm(True)
        k_norm = self._qk_norm(False)
        attn_qkv = qkv_quant + qkv_proj + q_norm + k_norm

        # rotary_embedding
        rotary_emb = self._rotary_emb()

        # core
        attn = self._attn_core()

        # o
        o_quant = self._attn_quant(False)
        o_proj = self._attn_proj(False)
        attn_o = o_quant + o_proj

        #norm
        post_attention_layernorm = self._norm()


        return pre_attention_layernorm, attn_qkv, rotary_emb, attn, attn_o, post_attention_layernorm

class Qwen3MoeSparseMoeBlock(torch.nn.Module):
    def __init__(self, args):
        super(Qwen3MoeSparseMoeBlock, self).__init__()
        self.args = args
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.ep = args.expert_model_parallel_size
        self.tp = args.tensor_model_parallel_size
        self.topk = args.num_experts_per_tok
    
    def _route_gate(self):
        # ReplicatedLinear
        x = torch.randn(self.batch_size, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(self.num_experts, self.hidden_size, device='cuda', dtype=torch.bfloat16)

        def test_func():
            F.linear(x, w)

        candidates = [
            ("cutlass", "cublasLt::splitK"),
            ("nvjet_tst", "cublasLt::splitK"),
            "cutlass",
            "nvjet_tst",
        ]
        last_err = None
        for names in candidates:
            try:
                res = bench_kineto(test_func, names, suppress_kineto_output=True)
                return sum(res) if isinstance(res, (list, tuple)) else float(res)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")       
        # kernel_names1 = ('cutlass', 'cublasLt::splitK')
        # kernel_names2 = ('nvjet_tst', 'cublasLt::splitK')
        try:
            t = sum(bench_kineto(test_func, kernel_names1, suppress_kineto_output=True))
        except:
                # H20 use a strange kernel named "nvjet_tst_xxx_v_bz_TNN" to perform
            t = sum(bench_kineto(test_func, kernel_names2, suppress_kineto_output=True))
        return t

    def _select_experts(self):
        # FusedMoE.select_experts
        seq_len = 1024 #TODO
        hidden_states = torch.randn(self.batch_size * seq_len, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        router_logits = torch.randn(self.batch_size * seq_len, self.num_experts, device='cuda', dtype=torch.bfloat16)

        def test_func():
            fused_topk(hidden_states=hidden_states,
                            gating_output=router_logits,
                            topk=self.topk,
                            renormalize=True)
        kernel_names = ('moe::topkGatingSoftmax',
                        'at::native::unrolled_elementwise_kernel',
                        'at::native::reduce_kernel',
                        'at::native::elementwise_kernel'
                        )
        t = sum(bench_kineto(test_func, kernel_names, suppress_kineto_output=True))
        return t

    def _moe_gate_proj(self, up_or_down):
        num_groups=self.num_experts // self.ep
        gbs = self.batch_size * (self.args.world_size // self.tp)
        m = gbs * self.topk // self.num_experts
        if up_or_down:
            n = 2 * self.args.moe_intermediate_size
            k = self.hidden_size
        else:
            n = self.hidden_size
            k = self.args.moe_intermediate_size

        def test_func():
            x_fp8, y_fp8, out, ref_out = construct_grouped(
                            num_groups, m, k, n, is_masked=True
                        )
            masked_m = torch.ones(
                        (num_groups,), device='cuda:0', dtype=torch.int) * m
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                        x_fp8, y_fp8, out, masked_m, m
                    )
        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)

        return t

    def _moe_act(self):
        # SiluAndMul
        num_groups=self.num_experts // self.ep
        gbs = self.batch_size * (self.args.world_size // self.tp)
        m = gbs * self.topk // self.num_experts
        n = 2 * self.args.moe_intermediate_size

        m_max = m # TODO: maybe should set in config
        intermediate_cache1 = torch.randn((num_groups * m_max, n),
                                        device='cuda',
                                        dtype=torch.bfloat16)
        intermediate_cache2 = torch.randn((num_groups * m_max, n // 2),
                                        device='cuda',
                                        dtype=torch.bfloat16)
        
        def test_func():
            torch.ops._C.silu_and_mul(
                intermediate_cache2.unflatten(0, (num_groups, m_max)),
                intermediate_cache1.unflatten(0, (num_groups, m_max)))
        # vllm::silu vllm::act_and_mul
        t = bench_kineto(test_func, "vllm::silu", suppress_kineto_output=True)
        return t


    def forward(self):
        route_gate = self._route_gate()
        route_select_experts = self._select_experts()
        moe_up = self._moe_gate_proj(True)
        moe_act = self._moe_act()
        moe_down = self._moe_gate_proj(False)

        return route_gate, route_select_experts, moe_up, moe_act, moe_down


class Qwen3MoeModel(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3MoeModel, self).__init__()
        self.time_list = {}
        self.args = args
        # self.Embedding = Qwen3Embedding(self.args)
        self.Attention = Qwen3MoeAttention(self.args)
        # self.MLP = Qwen3MoeMLP(self.args) # TODO add MLP only
        self.MoE = Qwen3MoeSparseMoeBlock(self.args)
    
    def forward(self):
        for i in range(self.args.num_hidden_layers):
            pre_norm, atten_qkv, atten_rotary_emb, atten_core, atten_o, post_norm = self.Attention()
            self.time_list.setdefault("atten_norm", []).append(
                        {"time_gpu": pre_norm* 1e6}
                    )
            self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv* 1e6}
                    )
            self.time_list.setdefault("atten_rotary_emb", []).append(
                        {"time_gpu": atten_rotary_emb* 1e6}
                    )
            self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core* 1e6}
                    )
            self.time_list.setdefault("atten_o", []).append(
                        {"time_gpu": atten_o* 1e6}
                    )
            self.time_list.setdefault("moe_norm", []).append(
                        {"time_gpu": post_norm* 1e6}
                    )
            moe_route_gate, moe_route_select_experts, moe_up, moe_act, moe_down = self.MoE()
            self.time_list.setdefault("moe_route_gate", []).append(
                        {"time_gpu": moe_route_gate* 1e6}
                    )
            self.time_list.setdefault("moe_route_select_experts", []).append(
                        {"time_gpu": moe_route_select_experts* 1e6}
                    )
            self.time_list.setdefault("moe_expert_up", []).append(
                        {"time_gpu": moe_up* 1e6}
                    )
            self.time_list.setdefault("moe_expert_act", []).append(
                        {"time_gpu": moe_act* 1e6}
                    )
            self.time_list.setdefault("moe_expert_down", []).append(
                        {"time_gpu": moe_down* 1e6}
                    )
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
    args = MockedQwen3Moe.Qwen3MoeParams()
    x = torch.randint(0, args.vocab_size, (2, 1))
    model = Qwen3MoeModel(args)
    filepath = model(x)
    print(filepath)     