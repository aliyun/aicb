# Lastest News
1. [2024/7] The first version of AICB is released. This version supports simulating workloads of GPT series, LLaMA, and Moe using the training frameworks of Megatron and DeepSpeed.

# Table of Contents

- [Lastest News](#lastest-news)
- [Table of Contents](#table-of-contents)
- [AICB Overview](#aicb-overview)
  - [Introduction](#introduction)
  - [The benchmark suite in AICB](#the-benchmark-suite-in-aicb)
- [Setup](#setup)
- [Usage](#usage)
  - [Running on physical GPU clusters](#running-on-physical-gpu-clusters)
    - [Basic parameters that you need to set](#basic-parameters-that-you-need-to-set)
    - [Running the whole benchmark suite](#running-the-whole-benchmark-suite)
    - [Running workloads for Megatron](#running-workloads-for-megatron)
    - [Running workloads for MOE](#running-workloads-for-moe)
    - [Running workloads for DeepSpeed](#running-workloads-for-deepspeed)
    - [Embedding the compuation patterns in the workload](#embedding-the-compuation-patterns-in-the-workload)
  - [Generate Workloads for Simulation (SimAI)](#generate-workloads-for-simulation-simai)
    - [Generating the workload description files for the whole benchmark suite](#generating-the-workload-description-files-for-the-whole-benchmark-suite)
    - [Generating the workload description files for Megatron](#generating-the-workload-description-files-for-megatron)
    - [Generating the workload description files for Moe](#generating-the-workload-description-files-for-moe)
    - [Generating the workload description files for DeepSpeed](#generating-the-workload-description-files-for-deepspeed)
  - [Running AICB with customized parameters](#running-aicb-with-customized-parameters)
    - [Running customized workloads on physical GPU clusters](#running-customized-workloads-on-physical-gpu-clusters)
    - [Generating customized workload description files](#generating-customized-workload-description-files)
- [Tutorial](#tutorial)
- [Projects using AICB](#projects-using-aicb)

# AICB Overview
## Introduction
AICB (Artificial Intelligence Communication Benchmark), is a novel benchmark suite for evaluating the communication system of a realistic and emulated GPU cluster from the pespectives of the emerging training and inference applications. Different from exisiting network benchmarks, AICB is designed to produce the communication workloads with precise patterns that are aligned to real-world applications. Taking the Large Language Model (LLM) training as an example, the workloads vary with the complicated combinations of models, parallel frameworks, and parameters of the models, parallel frameworks, and the collective communication libraries. In general, the scenarios suitable for using AICB include but not limited to 1) benchmarking and tuning of the communication system of a GPU cluster, 2) investigating and analyzing the communication patterns of specific application settings, 3) tools, e.g. simulators, that need workloads which are well described.

## The benchmark suite in AICB 
There are a lot of parameters that influence the communication and computation patterns, which are (1) model parameters (e.g., hidden_size, num_layers, seq_len, etc.) and (2) framework parameters (e.g., world size, parallelization strategies (TP, PP, DP, SP), zero level, reduce_bucket_size/allgather_bucket_size, etc.).
For the sake of generality, we cover those typical settings using a smallest set of benchmarks rather than traversing all the combinations. To this end, we propose the benchmark suite as listed in the following table.
**Users can directly run all the selected workloads selected in AICB, or run part of the workloads, or even generate their own workloads.**
For more detailed information, please refer to [AICB_workload spec v1.0](workload/Workload_spec_v1.0.csv).
| id  | Name      | Parameter_size | Hidden_size | Num_of_layers | Attention_heads | Sequence_length | FFN_hidden_size |
|-----|-----------|----------------|-------------|---------------|-----------------|-----------------|-----------------|
| 1   | GPT_7B    | 7B             | 4096        | 32            | 32              | 2048            | 16384           |
| 2   | GPT_13B   | 13B            | 5120        | 40            | 32              | 2048            | 20480           |
| 3   | GPT_22B   | 22B            | 6144        | 48            | 64              | 2048            | 24576           |
| 4   | GPT_175B  | 175B           | 12288       | 96            | 96              | 2048            | 49152           |
| 5   | GPT_13B   | 13B            | 5120        | 40            | 32              | 2048            | 20480           |
| 6   | LLaMA_7B  | 7B             | 4096        | 32            | 32              | 4096            | 11008           |
| 7   | LLaMA_7B  | 7B             | 4096        | 32            | 32              | 4096            | 11008           |
| 8   | LLaMA_65B | 65B            | 8192        | 80            | 64              | 4096            | 28672           |
| 9   | LLaMA_65B | 65B            | 8192        | 80            | 64              | 4096            | 28672           |
| 10   | Mistral_8*7B | 56B            | 4096        | 32            | 32              | 1024            | 14336           |


# Setup
You can follow the instrucitons below to quickly set up the environtments and run AICB.
1. Installation from source code

    a. To initiate actual communication tasks, ensure that the runtime environment has all necessary dependencies, such as CUDA and [PyTorch](https://pytorch.org), already installed. For specific usage examples, see [Physical Execution](#physical-execution)

    b. To generate workload traffic patterns for large model parallel framework training, you can use a CPU-only environment. For specific usage examples, see [Generate Workloads ](#generate-workloads-for-simulation-simai )

2. Installation from deb package (for Ubuntu systems)

    Currently, you can install the deb package  on an NV-built NGC container [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) to start running AICB.
    ```bash
    docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
    docker run --gpus all -it --rm -v /path/to/AICBench:/workspace/AICBench nvcr.io/nvidia/pytorch:xx.xx-py3
    dpkg -i /download/AICB_v1.0.deb 
    sh megatron_workload_with_aiob.sh -m 7
    ```

3. Composing a Docker image from Dockfile

    You can launch an instance of the Docker container  with Dockerfile for quick start:

    ```bash
    docker build -t image:latest .
    docker run --gpus all -it --rm image:latest 
    ```

# Usage
After installation, we provide three main usage scenarios for AICB:
1. [Running on physical GPU clusters](#running-on-physical-gpu-clusters) 
2. [Generating workload descrption files for simulation](#generate-workloads-for-simulation-simai) 
3. [Customized parameters](#customized-parameters).

There is a tutorial including all the details, please refer to [the tutorial](training/tutorial.md).
## Running on physical GPU clusters
For running AICB on a physical machine, we provide both [scripts](scripts/megatron_gpt.sh) for quick start and [methods](aicb.py) for executing custom cases.

### Basic parameters that you need to set
When running on a physical machine, additional configuration of environment variables required by PyTorch is necessary.
```bash
--nnodes                  Number of nodes: $WORLD_SIZE
--node_rank               Rank of the node: $RANK
--nproc_per_node          Number of GPUs per node: $NUM_GPUS
--master_addr             Master address: $MASTER_ADDR
--master_port             Master port: $MASTER_PORT
```

### Running the whole benchmark suite
You can directly execute all the test cases provided in our AICB workload specification v1.0 in physical GPU cluster by utilizing the [run_suites](run_suites.py) script. This script ensures that all parallel framworks are covered, allowing you to validate and analyze the performance and behavior of various workloads efficiently.

### Running workloads for Megatron
For the `Megatron parallel framework`, you can quickly start using the scripts/megatron_gpt.sh script file.
```bash
sh scripts/megatron_gpt.sh \
-m 7 --world_size 8 --tp_num 2 --pp_num 1 \
--comm_frame Megatron --global_batch 16  \
--micro_batch 1 --seq_length 2048
```

### Running workloads for MOE
For `Moe` , you can quickly start it using the [scripts/megatron_gpt.sh](scripts/megatron_gpt.sh) script file.
```bash
sh scripts/megatron_gpt.sh \
-m moe --world_size 8 --tp_num 2 --pp_num 1 
--moe_enabled --expert_parallel_size 1  \
--comm_frame Megatron --global_batch 16  \
--num_moe_experts 2 --moe_router_topk 2 \
--micro_batch 1  --grouped_gemm 
```

### Running workloads for DeepSpeed
For the `DeepSpeed` parallel framework, you can quickly start it using the [scripts/deepspeed_llama.sh](scripts/deepspeed_llama.sh) script file. Currently, the DeepSpeed framework does not support `--aiob_enable` or `--comp_filepath`, but you can choose to use fixed computation times (please refer to [the tutorial](training/tutorial.md)).
```bash
sh scripts/deepspeed_llama.sh \
--zero_stage 3 -m 65 --epoch_num 100 \
--reduce_bucket_size=1000000000 --allgather_bucket_size=500000000 \
--param_persistence_threshold=1000000 \
```
### Embedding the compuation patterns in the workload
To mirror the real-world workloads with both computation and communicaiton, we developed a sub-module, AIOB, that is used to generate computation patterns.
In AICB, we can enable AIOB to embed the computation time into the workloads.

For the Megatron parallel framework, the `--aiob_enable` option allows for capturing the computation time of each operation in the actual model. 
If we do not set `--aiob_enable`, only fixed computation times can be applied. (Please refer to [the tutorial](training/tutorial.md))

* Running workloads with computation times generated by AIOB. After running, we can get an extra computation desrcription file describing the computation times for the main computation kernels in the directory of `results/aiob_outputs`.
Note that the computation times are obtained through the execution of computation kernels on the specific GPU.
The following commands does not generate the computation descrition file, but also run the workload in the real GPU cluster.
```bash
sh scripts/megatron_gpt.sh \
-m 7 --world_size 8 --tp_num 2 --pp_num 1 \
--comm_frame Megatron --global_batch 16  \
--micro_batch 1 --seq_length 2048 \
--swiglu --use_flash_attn  --aiob_enable 
```
* Running workload with computation time through an existing computation decription file. 
Users can defined their own computation times or directly use the files we provided.
By specifying the computation description file with the `--comp_filepath` option, you can embed computation times before running the workload on a physical machine.
```bash
sh scripts/megatron_gpt.sh \
-m 7 --world_size 8 --tp_num 2 --pp_num 1 \
--comm_frame Megatron --global_batch 16  --micro_batch 1 \
--seq_length 2048 --swiglu --use_flash_attn  \
--aiob_enable  \
--comp_filepath workload/aiob_inputs/Example.txt
```
## Generate Workloads for Simulation (SimAI)
In addition to running the AICB in the GPU clusters, AICB also generates the workload description files which can be used for simulation or further analysis.
In this release, we provide [scripts](scripts/megatron_workload_with_aiob.sh) for quickly generating workloads for SimAI.

### Generating the workload description files for the whole benchmark suite
You can generate all the workload description files with [generate_suite]() as specified in our AICB workload spec v1.0. Once these files are created, you can execute them using the SimAI to test and analyze various scenarios.

### Generating the workload description files for Megatron
Here, you can use the script [scripts/megatron_workload.sh](scripts/megatron_workload_with_aiob.sh) and the parameter `--model_size` (7/13/22/175/moe) to generate the corresponding workload description file. For the computation part of the model, you can choose to enable AIOB by using the `--aiob_enable` option. If AIOB is not used, the Workload will be filled with a fixed computation time by default.
* Generating workload description files with computation times generated by AIOB.
```bash
sh ./scripts/megatron_workload_with_aiob.sh \
-m 7 --world_size 4096 \
--tp_num 2 --pp_num 1 \
--comm_frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 \
--swiglu --use_flash_attn  --aiob_enable
```
* Generating workload description files with computation time through an existing computation decription file. 

```bash
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tp_num 2 --pp_num 1 \
--comm_frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```
### Generating the workload description files for Moe
For the Moe, you can also use [scripts/megatron_workload_with_aiob.sh](scripts/megatron_workload_with_aiob.sh) to generate the corresponding model's workload file. 
```bash
sh scripts/workload_moe.sh \
-mmoe --world_size 4096 --tp_num 2 --pp_num 1 --sp  --expert_parallel_size 1 \
--num_moe_experts 2 --moe_router_topk 2  \
--comm_frame Megatron --global_batch 8192  \
--micro_batch 1 --seq_length 1024 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```

### Generating the workload description files for DeepSpeed
For the `DeepSpeed` parallel framework, you can use [scripts/workload_deepspeed.sh](scripts/deepspeed_llama.sh) to generate the corresponding workload description file.

```bash
sh ./scripts/deepspeed_llama.sh -m 7 
```

## Running AICB with customized parameters
In addition to quick start, you can also customize the model parameters in detail to run on physical clusters or generate the required workloads for simulation and analysis. For more detailed parameter descriptions and more Example, please refer to [the tutorial](training/tutorial.md).

### Running customized workloads on physical GPU clusters
The current entry file for running custom cases is [aicb.py](aicb.py). By using this file, you can flexibly choose more parameters for tuning.
```bash
# Megatron Example
torchrun \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--nproc_per_node gpu \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
./aicb.py --comm_frame=Megatron --world_size=$((WORLD_SIZE*8)) --tp_num=$tp_num \
  --micro_batch=$batch_size --global_batch=$((WORLD_SIZE*8*batch_size/tp_num)) --epoch_num=$epoch_num \
  --num_layers=$num_layers --hidden_size=$hidden_size --ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$num_attention_heads \
  $sp_enable --seq_len=$seq_len --vocab_size=$vocab_size --aiob_enable=$enable 
```
### Generating customized workload description files
Similarly, when generating workloads, you can also customize the model training parameters and modifying the generated files to generate your own workload file for simulation. This can be achieved by using the following files:
[generate custom description file](workload_generator/AIOB_simAI_workload_generator.py)

Here is an example:
```bash
python -m workload_generator.AIOB_simAI_workload_generator \
--world_size=32  --global_batch=64 --micro_batch=1 \
--num_layers=8 --num_attention_heads=176 --hidden_size=5120   \
--tp_num=2 --seq_length=4096 --swiglu --ffn_hidden_size=16384  \
--moe_router_topk=4  --enable_sequence_parallel --expert_parallel_size=16 \
--num_moe_experts=64 --moe_grouped_gemm --moe_enabled
```

# Tutorial
We provide a tutorial for users to quickly get started with AICB. [the tutorial](training/tutorial.md)

# Projects using AICB
Below are some of the projects where we have directly used AICB:
* AICB is part of the SimAI project which is led by Alibaba Cloud. Researchers who use AICB can cite our paper "SimAI: Unifying Architecture Design and Performance Tunning for Large-Scale Large Language Model Training with Scalability and Precision" (NSDI’25).

