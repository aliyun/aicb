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

#!/bin/python
"""example of running megatron on gpt-7B
python -m workload_generator.megatron_workload \
  --comm_frame=Megatron --world_size=16 --tp_num=8 --pp_num=1 --global_batch=64 --micro_batch=2 \
  --num_layers=32 --seq_length=2048 --hidden_size=4096 --epoch_num=2 --use-distributed-optimizer --enable_sequence_parallel
"""
from utils.utils import CommGroup, CommType, get_params, WorkloadWriter
import torch
from workload_generator.workload_generator import WorkloadGenerator
from workload_generator.mocked_model.MockedMegatron import MegatronModel
from log_analyzer.log import LogItem


class MegatronWorkload(WorkloadGenerator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.name = "megatron"
        self.args = args
        self.tp_is_enable = True if args.tp_num > 1 else False
        # print(f"total params: {self._get_total_params()}")

    def _get_total_params(self):
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
        return total_params

    def _get_layernorm_params(self):
        total_params = 0
        for param in self.model.parameters():
            if getattr(param, "sequence_parallel", False):
                total_params += param.numel()
        return total_params

    def init(self):
        args = self.args
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=1 * 8,
                stage="init.model_setup",
            )
        )
        for _ in range(3):
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=1 * 8,
                    stage="init.model_setup",
                )
            )
            if args.pp_num > 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=self.args.pp_num,
                        msg_size=1 * 8,
                        stage="init.model_setup",
                    )
                )
        # time
        self.workload.append(
            LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=4 * 8,
                stage="init.model_setup",
            )
        )

        self.workload.append(
            LogItem(
                comm_type=CommType.broadcast,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.args.tp_num,
                msg_size=3 * 8,
                stage="init.model_setup",
                src=0,
            )
        )

        if args.pp_rank == args.pp_num - 1 and args.pp_num > 1:
            for p in self.model.embedding.parameters():
                self.workload.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.args.tp_num,
                        msg_size=p.msg_size(),
                        stage="init.model_setup",
                    )
                )
        # time
        self.workload.append(
            LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=8 * 8,
                stage="init.model_setup",
            )
        )

    def get_pp_rank(self, rank, world_size, pp_size):
        ranks_per_pp_group = world_size // pp_size
        pp_rank = rank // ranks_per_pp_group
        return pp_rank

    def with_pipeline_forward_backward(self):
        args = self.args
        rank = torch.distributed.get_rank()
        world_size = args.world_size
        pp_rank = self.get_pp_rank(rank, world_size, args.pp_num)
        pp_num_warmup_microbatches = min(
            args.pp_num - pp_rank - 1, args.num_microbatches
        )
        num_microbatches_remaining = args.num_microbatches - pp_num_warmup_microbatches
        temp = self.model.forward()
        # forward_comm = self._get_comm_op(temp)

        for _ in range(pp_num_warmup_microbatches):
            if pp_rank != 0:
                # recv_prev
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="recv_prev",
                    )
                )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=5 * 8,
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),
                    stage="forward_step",
                    src=0,
                )
            )

            # for item in forward_comm:
            self.workload.extend(self.model.forward())

            if pp_rank != args.pp_num - 1:
                # send_next
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="send_next",
                    )
                )
        # recv prev
        if num_microbatches_remaining > 0 and pp_rank != 0:
            self.workload.append(
                LogItem(
                    comm_type=CommType.irecv,
                    comm_group=CommGroup.pp_group,
                    comm_group_size=1,
                    msg_size=2
                    * (args.hidden_size * args.seq_length * args.micro_batch),
                    stage="forward_step",
                    additional="recv_prev",
                )
            )

        for i in range(num_microbatches_remaining):
            last_iter = i == (num_microbatches_remaining - 1)
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=5 * 8,
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),
                    stage="forward_step",
                    src=0,
                )
            )

            self.workload.extend(self.model.forward())
            if pp_rank != args.pp_num - 1:
                # recv next
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="recv_next",
                    )
                )
                # send next
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="send_next",
                    )
                )

            self.workload.extend(self.model.backward())

            if pp_rank != 0:
                if last_iter:
                    # send prev
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.isend,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2
                            * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="send_prev",
                        )
                    )
                else:
                    # send prev recv prev
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.isend,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2
                            * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="send_prev",
                        )
                    )
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.irecv,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2
                            * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="recv_prev",
                        )
                    )

        for _ in range(pp_num_warmup_microbatches):
            # recv next
            if pp_rank != args.pp_num - 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="backward_step",
                        additional="recv_next",
                    )
                )

            self.workload.extend(self.model.backward())

            # send prev
            if pp_rank != 0:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2
                        * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="backward_step",
                        additional="send_prev",
                    )
                )

    def forward(self):
        args = self.args
        if self.tp_is_enable:
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=5 * 8,
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),
                    stage="forward_step",
                    src=0,
                )
            )
        self.workload.extend(self.model.forward())
        for _ in range(3):
            # for bf16, we need to use float32 in loss communication
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tp_num,
                    msg_size=args.micro_batch * args.seq_length * 4,
                    stage="forward_step._VocabParallelCrossEntropy",
                )
            )
        # average_losses_across_data_parallel_group
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=1 * 4,
                stage="forward_step.average_losses_across_data_parallel_group",
            )
        )

    def backward(self):
        self.workload.extend(self.model.backward())

    def step(self):
        args = self.args

        if args.use_distributed_optimizer:
            self.workload.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=4 * self._get_total_params() // (args.pp_num),
                    stage="step",
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=2 * self._get_total_params() // (args.pp_num),
                    stage="step",
                )
            )
        else:
            # 注意，如果使用过了bf16，那么梯度会使用tf32
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=4 * self._get_total_params() // (args.pp_num),
                    stage="step.finish_grad_sync",
                )
            )

        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.args.tp_num,
                msg_size=2 * self._get_layernorm_params() // (args.pp_num),
                stage="step._allreduce_layernorm_grads",
            )
        )

        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.args.tp_num,
                msg_size=4,
                stage="step.check_for_nan",
            )
        )


if __name__ == "__main__":
    args = get_params()
    model = MegatronModel(args)
    workload_generator = MegatronWorkload(args, model)
    workload = workload_generator()
    filename = f"results/mocked_workload/pp_rank{args.pp_rank}_local_megatron.csv"
    workload.dump(args, filename)
    # WorkloadWriter.write_workload(workload, args, filename)
