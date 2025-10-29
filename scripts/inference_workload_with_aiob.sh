SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_size=deepseek-671B
config_file_path=
dpsk_default_path="$SCRIPT_DIR/inference_configs/deepseek_default.json"
qwen3_moe_default_path="$SCRIPT_DIR/inference_configs/qwen3_moe_default.json"
qwen3_next_default_path="$SCRIPT_DIR/inference_configs/qwen3_next_default.json"

usage() {
  echo "Usage: \$0 [options]
    options:
      -m, --model_size          model size, defaults to $model_size (possible values: deepseek-671B, qwen3-235B)
      -c, --config              config file path, default is None
      -h, --help                display this help and exit" 1>&2; exit 1;
}

while [ $# -gt 0 ]
do
  case $1 in
    -m|--model_size)
      model_size=$2; shift;;
    -c|--config)
      config_file_path=$2; shift;;
    -h|--help)
      usage;;
    (*)
      break;;
  esac
  shift
done

case $model_size in
  deepseek-671B)
    model_name=DeepSeek-671B
    config_file_path=${config_file_path:-$dpsk_default_path}
    ;;
  qwen3-235B)
    model_name=Qwen3-Moe-235B
    config_file_path=${config_file_path:-$qwen3_moe_default_path}
    ;;
  qwen3-next-80B)
    model_name=Qwen3-Next-80B
    config_file_path=${config_file_path:-$qwen3_next_default_path}
    ;;
  (*)
    break;;
esac

cmd="python -m workload_generator.SimAI_inference_workload_generator $model_name $config_file_path"

echo $cmd

$cmd
