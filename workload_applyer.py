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
import sys
import math
import time
from utils.utils import WorkloadWriter, CommGroup, CommType, ReduceOp
from utils.benchmark_logger import bench_logger
import utils.utils as utils


class WorkloadApplyer:
    def __init__(self, workload=None, args=None, filename=None) -> None:
        if workload is None or args is None:
            assert (
                filename is None
            ), f"you should either pass workload,args or filename to init WorkloadApplyer"
            workload, args = WorkloadWriter.load_workload(filename)
        # if not hasattr(args, "backend"):
        #     args.backend = "nccl"
        # torch.distributed.init_process_group(backend=args.backend)
        self.args = args
        world_size = torch.distributed.get_world_size()
        # args.rank = torch.distributed.get_rank()
        if args.world_size != world_size:
            print(
                f"WARNNING: world_size is {args.world_size} when generating workload, but now world size is {world_size}"
            )
            args.world_size = torch.distributed.get_world_size()
        device_count = torch.cuda.device_count()
        self.device = args.rank % device_count
        torch.cuda.set_device(self.device)
        self.device = torch.cuda.current_device()
        self.comm_group_info, self.pp_global_rank_info = (
            self._generate_dp_tp_pp_groups()
        )
        self.workload = workload
        self.comm_type_function = {
            CommType.barrier: self._apply_barrier,
            CommType.broadcast: self._apply_broadcast,
            CommType.reduce: self._apply_reduce,
            CommType.all_reduce: self._apply_all_reduce,
            CommType.all_gather: self._apply_all_gather,
            CommType.reduce_scatter: self._apply_reduce_scatter,
            CommType.isend: self._apply_p2pcommunication,
            CommType.irecv: self._apply_p2pcommunication,
            CommType.all_gather_into_tensor: self._apply_all_gather,
            CommType.reduce_scatter_tensor: self._apply_reduce_scatter,
            CommType.computation: self._apply_computation,
            CommType.all_to_all: self._apply_all_to_all,
            CommType.epoch_end: bench_logger.end_epoch,
        }
        cal_tuple_num = lambda t: math.prod(t[0]) + math.prod(t[1])
        max_msg_size = max(
            [
                (
                    item.msg_size
                    if isinstance(item.msg_size, int)
                    else cal_tuple_num(item.msg_size)
                )
                for item in self.workload.workload
            ]
        )
        self.gemm_cache = {}
        self.computation_aiob = False
        if args.aiob_enable:
            self.computation_aiob = True

        self.skip_computation = False
        self.always_apply_gemm = False
        self.gemm_iters = 1 if self.always_apply_gemm else 50
        self.buffer = torch.empty(
            (max_msg_size,), dtype=torch.bfloat16, device=self.device
        )

    def _generate_dp_tp_pp_groups(self):
        all_data_parallel_group_ranks = []
        world_size = self.args.world_size
        rank = torch.distributed.get_rank()
        self.rank = rank
        tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size = (
            self.args.tp_num,
            self.args.pp_num,
            self.args.dp_num,
        )
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = (
            world_size // pipeline_model_parallel_size
        )
        num_data_parallel_groups: int = world_size // data_parallel_size

        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    dp_group = group
        for i in range(data_parallel_size):
            ranks = [
                data_parallel_group_ranks[i]
                for data_parallel_group_ranks in all_data_parallel_group_ranks
            ]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                mp_group = group
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(
                i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
            )
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                tp_group = group
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                pp_global_rank = ranks
                pp_group = group
        return {
            CommGroup.tp_group: tp_group,
            CommGroup.dp_group: dp_group,
            CommGroup.pp_group: pp_group,
        }, pp_global_rank

    def _get_pipeline_parallel_size(self):
        group = self.comm_group_info["pp_group"]
        pp_group_size = torch.distributed.get_world_size(group)
        return pp_group_size

    def _get_pipeline_parallel_rank(self):
        group = self.comm_group_info["pp_group"]
        pp_rank = torch.distributed.get_rank(group)
        return pp_rank

    def _get_pipeline_prev_rank(self):
        rank_in_pipeline = self._get_pipeline_parallel_rank()
        world_size = self._get_pipeline_parallel_size()
        return self.pp_global_rank_info[(rank_in_pipeline - 1) % world_size]

    def _get_pipeline_next_rank(self):
        rank_in_pipeline = self._get_pipeline_parallel_rank()
        world_size = self._get_pipeline_parallel_size()
        return self.pp_global_rank_info[(rank_in_pipeline + 1) % world_size]

    def _apply_p2pcommunication(self, item):
        ops = []
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2)
        if item.additional == "send_prev":
            if self._get_pipeline_parallel_rank() != 0:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor, self._get_pipeline_prev_rank()
                )
                ops.append(send_prev_op)
            else:
                pass
        if item.additional == "send_next":
            if self._get_pipeline_parallel_rank() != self.args.pp_num - 1:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor, self._get_pipeline_next_rank()
                )
                ops.append(send_next_op)
            else:
                pass
        if item.additional == "recv_prev":
            if self._get_pipeline_parallel_rank() != 0:
                tensor_recv_prev = torch.empty(
                    item.msg_size // 2, dtype=torch.bfloat16, device=self.device
                )
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor_recv_prev,
                    self._get_pipeline_prev_rank(),
                )
                ops.append(recv_prev_op)
            else:
                pass
        if item.additional == "recv_next":
            if self._get_pipeline_parallel_rank() != self.args.pp_num - 1:
                tensor_recv_next = torch.empty(
                    item.msg_size // 2, dtype=torch.bfloat16, device=self.device
                )
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor_recv_next,
                    self._get_pipeline_next_rank(),
                )
                ops.append(recv_next_op)
            else:
                pass
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        torch.cuda.synchronize()

    def _apply_barrier(self, item):
        torch.distributed.barrier()

    @bench_logger.log_timing("comm")
    def _apply_broadcast(self, item):
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2)
        group = self.comm_group_info[item.comm_group]
        src = torch.distributed.get_global_rank(group, 0)
        return torch.distributed.broadcast(
            tensor=tensor, src=src, group=group, async_op=False
        )

    @bench_logger.log_timing("comm")
    def _apply_reduce(self, item):
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2)
        group = self.comm_group_info[item.comm_group]
        dst = item.dst
        return torch.distributed.reduce(
            tensor=tensor,
            dst=dst,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
            async_op=False,
        )

    @bench_logger.log_timing("comm")
    def _apply_all_reduce(self, item):
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2)
        group = self.comm_group_info[item.comm_group]
        return torch.distributed.all_reduce(
            tensor=tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
            async_op=False,
        )

    @bench_logger.log_timing("comm")
    def _apply_all_gather(self, item):
        group = self.comm_group_info[item.comm_group]
        num_elements = item.msg_size // 2
        padding_size = (
            (group.size() - num_elements % group.size())
            if num_elements % group.size()
            else 0
        )
        num_elements = num_elements + padding_size
        output_tensor = torch.narrow(self.buffer, 0, 0, num_elements)
        input_tensor_size = output_tensor.numel() // group.size()
        group_rank = torch.distributed.get_group_rank(group, self.rank)
        input_tensor = torch.narrow(
            output_tensor, 0, group_rank * input_tensor_size, input_tensor_size
        )
        return torch.distributed.all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=False
        )

    @bench_logger.log_timing("comm")
    def _apply_reduce_scatter(self, item):
        group = self.comm_group_info[item.comm_group]
        num_elements = item.msg_size // 2
        padding_size = (
            (group.size() - num_elements % group.size())
            if num_elements % group.size()
            else 0
        )
        num_elements = num_elements + padding_size
        input_tensor = torch.narrow(self.buffer, 0, 0, num_elements)
        group = self.comm_group_info[item.comm_group]
        output_tensor_size = input_tensor.numel() // group.size()
        group_rank = torch.distributed.get_group_rank(group, self.rank)
        output_tensor = torch.narrow(
            input_tensor, 0, group_rank * output_tensor_size, output_tensor_size
        )
        return torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=group, async_op=False
        )

    @bench_logger.log_timing("comm")
    def _apply_all_to_all(self, item):
        group = self.comm_group_info[item.comm_group]
        num_elements = item.msg_size // 2
        input_tensor = torch.narrow(self.buffer, 0, 0, num_elements)
        # output_tensor = torch.narrow(self.buffer, 0, 0 , num_elements)
        output_tensor = torch.empty(
            num_elements * group.size(),
            dtype=self.buffer.dtype,
            device=self.buffer.device,
        )
        return torch.distributed.all_to_all_single(
            output_tensor, input_tensor, group=group
        )

    @bench_logger.log_timing("comp")
    def _apply_computation(self, item):
        if self.skip_computation:
            return
        if self.computation_aiob:
            time.sleep(item.msg_size / 1e9)
        else:
            if self.always_apply_gemm or item.msg_size not in self.gemm_cache:
                input_shape1, input_shape2 = item.msg_size
                A, B = torch.rand(input_shape1, device=self.device), torch.rand(
                    input_shape2, device=self.device
                )
                t = time.time()
                for _ in range(self.gemm_iters):
                    torch.matmul(A, B)
                self.gemm_cache[item.msg_size] = (time.time() - t) / self.gemm_iters
            else:
                time.sleep(self.gemm_cache[item.msg_size])

    def apply_workload(self):
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        key = "backward"
        for item in self.workload.workload:
            if (
                self.computation_aiob
                and item.comm_type == CommType.all_reduce
                and key in item.stage
            ):
                comm_func = self.comm_type_function[item.comm_type]
                # comm_func(item)
            else:
                comm_func = self.comm_type_function[item.comm_type]
                comm_func(item)
        torch.cuda.synchronize(self.device)
        end = time.perf_counter()
        return end - start


if __name__ == "__main__":
    filename = "results/model_workload/local_deepspeed_stage3.csv"
    applyer = WorkloadApplyer(filename=filename)
    applyer.apply_workload()
    # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if torch.distributed.get_rank() == 0:
        bench_logger.analyze_comm_log(bench_logger.comm_log)
