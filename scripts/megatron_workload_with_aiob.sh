#!/bin/sh


comm_frame=Megatron
world_size=32
tp_num=8
pp_num=1
global_batch=1024
micro_batch=1
num_layers=40
seq_length=4096
hidden_size=5140
epoch_num=1
num_attention_heads=40
aiob_enable=
use_flash_attn=
swiglu=
sp_enable=
ffn_hidden_size=
comp_filepath=
model_size=13
max_position_embeddings=4096
vocab_size=50257
usage() {
  echo "Usage: \$0 [options]
    options:
      --comm_frame              communication framework, defaults to $comm_frame
      --world_size              world size, defaults to $world_size
      --tp_num                  tensor parallelism size, defaults to $tp_num
      --pp_num                  pipeline parallelism size, defaults to $pp_num
      --global_batch            global batch size, defaults to $global_batch
      --micro_batch             micro batch size, defaults to $micro_batch
      --num_layers              number of layers, defaults to $num_layers
      --seq_length              sequence length, defaults to $seq_length
      --hidden_size             hidden size, defaults to $hidden_size
      --epoch_num               number of epochs, defaults to $epoch_num
      --use_distributed_optimizer use distributed optimizer
      --num_attention_heads     number of attention heads, defaults to $num_attention_heads
      --aiob_enable             enable AIOB
      --use_flash_attn          use flash attention
      --swiglu                  use swiglu
      --ffn_hidden_size         FFN hidden size
      --comp_filepath           computation file path
      --max_position_embeddings max position embeddings, defaults to $max_position_embeddings
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7, moe)
      --moe_enabled             enable moe
      --moe_router_topk         Number of experts to route to for each token.
      --expert_parallel_size     Degree of expert model parallelism
      --num_moe_experts          Number of experts in the MoE model.  
      --moe_grouped_gemm        apply grouped gemm
      -h, --help                display this help and exit" 1>&2; exit 1;
}


while [ $# -gt 0 ]
do
  case $1 in
    --comm_frame)
      comm_frame=$2; shift;;
    --world_size)
      world_size=$2; shift;;
    --tp_num)
      tp_num=$2; shift;;
    --pp_num)
      pp_num=$2; shift;;
    --global_batch)
      global_batch=$2; shift;;
    --micro_batch)
      micro_batch=$2; shift;;
    --num_layers)
      num_layers=$2; shift;;
    --seq_length)
      seq_length=$2; shift;;
    --hidden_size)
      hidden_size=$2; shift;;
    --epoch_num)
      epoch_num=$2; shift;;
    --num_attention_heads)
      num_attention_heads=$2; shift;;
    --aiob_enable)
      aiob_enable=--aiob_enable;;
    --use_flash_attn)
      use_flash_attn=--use_flash_attn;;
    --swiglu)
      swiglu=--swiglu;;
    --ffn_hidden_size)
      ffn_hidden_size=$2; shift;;
    --sp|--sp-enable)
      sp_enable=--enable_sequence_parallel;;
    --comp_filepath)
      comp_filepath=$2; shift;;
    -m|--model_size)
      model_size=$2; shift;;
    --max_position_embeddings)
      max_position_embeddings=$2 ; 
      shift;;
    --moe_enabled)
      moe_enabled=--moe_enabled;;
    --moe_router_topk)
      moe_router_topk=$2; shift;;
    --num_moe_experts)
      num_moe_experts=$2; shift;;
    --expert_parallel_size)
      expert_parallel_size=$2; shift;;
    --grouped_gemm)
      grouped_gemm=--moe_grouped_gemm;;
    -h|--help)
      usage;;
    (*)
      break;;
  esac
  shift
done


case $model_size in
  175)
    model_name=gpt_175B
    num_layers=96
    hidden_size=12288
    num_attention_heads=96
    tp_num=8
    ;;
  22)
    model_name=gpt_22B
    num_layers=48
    hidden_size=6144
    num_attention_heads=64
    tp_num=8
    ;;
  13)
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
  7)
    model_name=gpt_7B
    num_layers=36
    hidden_size=4096
    num_attention_heads=32
    tp_num=4
    ;;
  moe)
    model_name=Mixtral_8*7B
    num_layers=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    tp_num=2
    ;;
  (*)
    echo "Only support model size 175, 22,13 or 7; using default size 13"
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac


cmd="python -m workload_generator.AIOB_simAI_workload_generator \
  --comm_frame=$comm_frame \
  --world_size=$world_size \
  --tp_num=$tp_num \
  --pp_num=$pp_num \
  --global_batch=$global_batch \
  --micro_batch=$micro_batch \
  --num_layers=$num_layers \
  --seq_length=$seq_length \
  --hidden_size=$hidden_size \
  --epoch_num=$epoch_num \
  --num_attention_heads=$num_attention_heads \
  --model_name=$model_name \
  --max_position_embeddings=$max_position_embeddings \
  --vocab_size=$vocab_size \
  --use-distributed-optimizer
  ${aiob_enable} \
  ${use_flash_attn} \
  ${swiglu} \
  ${sp_enable} \
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \
  ${comp_filepath:+--comp_filepath=$comp_filepath} \
  ${moe_enabled} \
  ${moe_router_topk:+--moe_router_topk=$moe_router_topk} \
  ${num_moe_experts:+--num_moe_experts=$num_moe_experts} \
  ${expert_parallel_size:+--expert_parallel_size=$expert_parallel_size} \
  ${grouped_gemm} " \

echo $cmd


$cmd
