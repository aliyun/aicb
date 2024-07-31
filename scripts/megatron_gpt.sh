#!/bin/sh

set -x
: ${WORLD_SIZE:=1}
: ${RANK:=0}
: ${MASTER_ADDR:="localhost"}
: ${MASTER_PORT:=29500}
NUM_GPUS=8
model_size=13
num_layers=40
num_attention_heads=40
hidden_size=5120
seq_len=2048
# micro_batch=2
epoch_num=1
tp_num=8
pp_num=1
vocab_size=50257
model_name=gpt_13b
global_batch=1024
sp_enable=
comm_frame=Megatron
aiob_enable=1
max_position_embeddings=4096


usage() {
  echo "Usage: \$0 [options]
    options:
      --comm_frame              Communication framework: $comm_frame
      --world_size              World size (number of nodes): $WORLD_SIZE
      --tp_num                  Tensor parallelism size: $tp_num
      --pp_num                  Pipeline parallelism size: $pp_num
      --global_batch            Global batch size: $global_batch
      --micro_batch             Micro batch size: $micro_batch
      --num_layers              Number of layers: $num_layers
      --seq_length              Sequence length: $seq_length
      --hidden_size             Hidden size: $hidden_size
      --epoch_num               Number of epochs: $epoch_num
      --num_attention_heads     Number of attention heads: $num_attention_heads
      --aiob_enable             Enable AIOB: $aiob_enable
      --use_flash_attn          Use flash attention: $use_flash_attn
      --swiglu                  Use SWIGLU: $swiglu
      --ffn_hidden_size         FFN hidden size: $ffn_hidden_size
      --comp_filepath           Computation file path: $comp_filepath
      --model_name              Model name: $model_name
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7)
      --max_position_embeddings Max position embeddings: $max_position_embeddings
      --nnodes                  Number of nodes: $WORLD_SIZE
      --node_rank               Rank of the node: $RANK
      --nproc_per_node          Number of GPUs per node: $NUM_GPUS
      --master_addr             Master address: $MASTER_ADDR
      --master_port             Master port: $MASTER_PORT
      -h, --help                Display this help and exit"1>&2; exit 1;
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
    --nnodes)
      WORLD_SIZE=$2;shift;;
    --node_rank)
      RANK=$2;shift;;
    --nproc_per_node)
      NUM_GPUS=$2;shift;;
    --master_addr)
      MASTER_ADDR=$2;shift;;
    --master_port)
      MASTER_PORT=$2;shift;;  
    -h|--help)
      usage ;;
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
  (*)
    echo "Only support model size 175, 22,13 or 7; using default size 13"
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac

script="./aicb.py"
which numarun
if [[ $? -eq 0 ]]; then
    script="--no-python numarun python ./aicb.py"
fi

cmd="$script \
  --comm_frame=$comm_frame \
  --model_name=$model_name \
  --world_size=$(($WORLD_SIZE * $NUM_GPUS)) \
  --tp_num=$tp_num \
  --micro_batch=$micro_batch \
  --global_batch=$global_batch \
  --epoch_num=$epoch_num \
  --num_layers=$num_layers \
  --hidden_size=$hidden_size \
  --num_attention_heads=$num_attention_heads \
  --seq_length=$seq_length \
  --vocab_size=$vocab_size \
  --pp_num=$pp_num \
  --use-distributed-optimizer \
  --max_position_embeddings=$max_position_embeddings \
  ${aiob_enable} \
  ${sp_enable} \
  ${use_flash_attn} \
  ${swiglu} \
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \
  ${comp_filepath:+--comp_filepath=$comp_filepath}"
echo $cmd


torchrun \
  --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --nproc_per_node $NUM_GPUS \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  $cmd
