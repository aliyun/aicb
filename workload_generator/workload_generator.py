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

from workload_generator.mocked_model.MockedModel import MockedModel
from utils.utils import CommGroup, CommType
from log_analyzer.log import Workload, LogItem


class WorkloadGenerator:
    # generator = WorkloadGenerator
    def __init__(self, args, model: MockedModel) -> None:
        self.name = "workload_generator"
        self.args = args
        self.model = model
        self.workload = Workload()
        self.epoch = 0

    def __call__(self):
        args = self.args
        self.workload = Workload()
        self.init()
        self.workload.append(LogItem(comm_type=CommType.epoch_end))
        for i in range(args.epoch_num):
            if args.pp_num > 1 and args.comm_frame != "collective_test":
                self.with_pipeline_forward_backward()
                self.step()
            else:
                for _ in range(args.num_microbatches):
                    self.forward()
                    self.backward()
            self.step()
            self.workload.append(LogItem(comm_type=CommType.epoch_end))
        return self.workload

    def forward(self):
        pass

    def backward(self):
        pass

    def step(self):
        pass

    def with_pipeline_forward_backward(self):
        pass
