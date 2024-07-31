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
import sys
import subprocess
import os
import configparser
import argparse

running_command = {}
default_conifg = {
    "deepspeed": {
        "llama7b_zero2": 1,
        "llama7b_zero3": 0,
        "llama65b_zero2": 0,
        "llama65b_zero3": 1,
    },
    "megatron": {
        "llama_7B": 1,
        "gpt_13B_tp": 1,
        "gpt_13B_sp": 1,
        "llama_65B": 1,
        "gpt_175B": 1,
    },
    "coll_comm_check": {"all_reduce": 1, "all_gather": 1, "muti_all_reduce": 1},
}


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output directory", default="./results")


def read_config(config):
    ds_conf = config["deepspeed"]
    megatron_conf = config["megatron"]
    cc_conf = config["coll_comm_check"]
    if int(ds_conf["llama7b_zero2"]):
        running_command["deepspeed2_llama7b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 2 -m 7 --epoch_num 100"
        )
    if int(ds_conf["llama7b_zero3"]):
        running_command["deepspeed3_llama7b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 3 -m 7 --epoch_num 100 \
      --reduce_bucket_size=5000000000 --allgather_bucket_size=5000000000 \
      --param_persistence_threshold=40960",
        )
    if int(ds_conf["llama65b_zero2"]):
        running_command["deepspeed2_llama65b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 2 -m 65 --epoch_num 100 \
          --reduce_bucket_size=1000000000 --allgather_bucket_size=500000000 \
          --param_persistence_threshold=1000000"
        )
    if int(ds_conf["llama65b_zero3"]):
        running_command["deepspeed3_llama65b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 3 -m 65 --epoch_num 100 \
          --reduce_bucket_size=1000000000 --allgather_bucket_size=500000000 \
          --param_persistence_threshold=1000000"
        )
    if int(megatron_conf["llama_7B"]):
        running_command["megatron_llama7B"] = (
            f"bash scripts/megatron_gpt.sh -m 7 --tp_num 1 --epoch_num 100"
        )
    if int(megatron_conf["gpt_13B_tp"]):
        running_command["megatron_gpt13b_tp"] = (
            f"bash scripts/megatron_gpt.sh -m 13 --tp_num 4 --epoch_num 100"
        )
    if int(megatron_conf["gpt_13B_sp"]):
        running_command["megatron_gpt13b_sp"] = (
            f"bash scripts/megatron_gpt.sh -m 13 --tp_num 4 --epoch_num 100 --sp"
        )
    if int(megatron_conf["llama_65B"]):
        running_command["megatron_llama65B"] = (
            f"bash scripts/megatron_gpt.sh -m 65 --tp_num 8 --epoch_num 100"
        )
    if int(megatron_conf["gpt_175B"]):
        running_command["megatron_gpt175B"] = (
            f"bash scripts/megatron_gpt.sh -m 175 --tp_num 8 --epoch_num 100  --pp_num 2"
        )
    if int(cc_conf["all_reduce"]):
        running_command["all_reduce_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_reduce --model_name all_reduce"
        )
    if int(cc_conf["all_gather"]):
        running_command["all_gather_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_gather --model_name all_gather"
        )
    if int(cc_conf["muti_all_reduce"]):
        running_command["muti_all_reduce_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_reduce --model_name muti_all_reduce --muti_all_reduce_enable 1"
        )


if __name__ == "__main__":
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    read_config(config=default_conifg)
    result = {}
    print(running_command)
    for name, command in running_command.items():
        ret = subprocess.run(command, shell=True, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"ERROR when running {name}: {command}")
            print(
                f"return state is {ret.returncode}, got err{ret.stderr}, get output{ret.stdout}"
            )
            exit(-1)
        command_out = ret.stdout
        print(command_out)
