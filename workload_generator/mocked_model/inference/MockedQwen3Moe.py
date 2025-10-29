from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem
#TODO support Workload

class Qwen3MoeRMSNorm(MockedModel):
    def __init__(self,
        layerid,
        prefix_name):
        self.name = prefix_name + "norm"
        self.layer_id = layerid

class Qwen3MoeAttention(MockedModel):
    def __init__(self,
                 layerid):
        self.name = "attention_layer"
        self.layer_id = layerid

class Qwen3MoeRoute(MockedModel):
    def __init__(self,
                 layerid):
        self.name = "moe_route"
        self.layer_id = layerid

class Qwen3MoeExpert(MockedModel):
    def __init__(self,
                 layerid):
        self.name = "moe_expert"
        self.layer_id = layerid

class Qwen3MoeBlock(MockedModel):
    def __init__(self,
                 layerid):
        self.name = "moe_block"
        self.layer_id = layerid
        self.route = Qwen3MoeRoute(layerid)
        self.moeGemm = Qwen3MoeExpert(layerid)

class Qwen3MoeTransformerLayer(MockedModel):
    def __init__(self,
                 layerid
                 ):
        self.pre_norm = Qwen3MoeRMSNorm(layerid,prefix_name="attention_")
        self.attention = Qwen3MoeAttention(layerid)
        self.post_norm = Qwen3MoeRMSNorm(layerid,prefix_name="moe_")
        self.MoE = Qwen3MoeBlock(layerid)


class Qwen3MoeModel(MockedModel):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.num_hidden_layers = config.num_hidden_layers
        self.moe_intermediate_size = config.moe_intermediate_size
        self.layers = [
            Qwen3MoeTransformerLayer(
                i
            )
            for i in range(config.num_hidden_layers)    
        ]
        # print(config)

class Qwen3MoeParams():
    def __init__(self, config_file=None):
        # Default values
        self.aiob_enable: bool = True
        self.model_name: str = "Qwen3-Moe-235B"
        self.frame: str = "Qwen3-Moe"

        # Input Params：
        self.seq_length: int = 1  # decode
        self.micro_batch: int = 64  # for example
        self.world_size: int = 32  # Total 32 gpus
        self.tensor_model_parallel_size: int = 1
        self.expert_model_parallel_size: int = 32
        self.pipeline_model_parallel: int = 1
        
        self.moe_enable = True

        self.result_dir = "results/workload/"

        # Load from config file if provided
        if config_file:
            self.load_from_config(config_file)

    def load_from_config(self, config_file):
        import json
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update attributes with values from config file
            for key, value in config_data.items():
                # print(key,value)
                # if hasattr(self, key):
                setattr(self, key, value)
            
            # Recalculate total_experts if needed
            if 'router_expert' in config_data or 'duped_expert' in config_data:
                self.total_experts = self.router_expert + self.duped_expert
                
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default values.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {config_file}. Using default values.")


if __name__ == "__main__":
    import sys
    # Check if a config file is provided as a command line argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    args = Qwen3MoeParams(config_file)
    model = Qwen3MoeModel(args)
    # workloads = model.forward(args)
    # filename = "testqwen3-1.csv" #TODO: add forward
    # workloads.dump(filename)
    # print("Finish Model initialization")