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
import torch
from utils.utils import get_args, get_comp_out, extract_averages, Comp_with_aiob
from utils.benchmark_logger import bench_logger
from workload_generator.mocked_model.MockedDeepspeed import DeepspeedForCausalLM
from workload_generator.mocked_model.MockedMegatron import MegatronModel
from workload_generator.generate_deepspeed_stage1_2_workload import (
    DeepSpeedStage1,
    DeepSpeedStage2,
)
from workload_generator.generate_deepspeed_stage3_workload import DeepSpeedStage3
from workload_generator.generate_megatron_workload import MegatronWorkload
from workload_generator.generate_collective_test import Collective_Test
from workload_applyer import WorkloadApplyer

if __name__ == "__main__":
    args = get_args()
    if not hasattr(args, "backend"):
        args.backend = "nccl"
    torch.distributed.init_process_group(backend=args.backend)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    if args.comm_frame == "Megatron":
        model = MegatronModel(args)
        workload_generator = MegatronWorkload(args, model)
    elif args.comm_frame == "DeepSpeed":
        model = DeepspeedForCausalLM(args)
        if args.stage == 1:
            workload_generator = DeepSpeedStage1(args, model)
        elif args.stage == 2:
            workload_generator = DeepSpeedStage2(args, model)
        elif args.stage == 3:
            workload_generator = DeepSpeedStage3(args, model)
    elif args.comm_frame == "collective_test":
        workload_generator = Collective_Test(args, None)
    workload = workload_generator()
    if args.aiob_enable:
        params = model.parameters()
        args.model_param = sum(p.numel() for p in params)
        if args.comp_filepath == None:

            filepath = get_comp_out(args)
            compute_cache = extract_averages(filepath)
        else:
            print("comp_filepath:", args.comp_filepath)
            compute_cache = extract_averages(args.comp_filepath)
        workload = Comp_with_aiob(workload, compute_cache)
    applyer = WorkloadApplyer(workload=workload, args=args)
    if torch.distributed.get_rank() == 0:
        filename = f"{workload_generator.name}_{args.model_name}_{args.world_size}n.csv"
        workload.dump(args, filename)
        # WorkloadWriter.write_workload(workload, args, filename)
    cpu_time = applyer.apply_workload()
    if torch.distributed.get_rank() == 0:
        bench_logger.analyze_comm_log()
        if args.comm_frame != "collective_test":
            bench_logger.analyze_comm_time()
        bench_logger.dump_log(filename)
        print(
            f"total time for {args.comm_frame} and {args.epoch_num} epochs is {cpu_time:.4f} s"
        )
