import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=8 dtest_xformer_mlp.py
"""

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 1024, bias=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.relu(out1)
        out3 = self.fc2(out2)
        return out3

mesh = init_device_mesh("cuda", (8,))
rank = dist.get_rank()

def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)


# Trace the module
#print(sharded_module)
#trace = torch.fx.symbolic_trace(sharded_module)
#print(trace.graph)

inputs = torch.randn(16, 512, 1024, device='cuda')
dinp = distribute_tensor(inputs, mesh, [Replicate()])

out = sharded_module(dinp)

# Check memory consumption
memory_allocated = torch.cuda.memory_allocated()
if rank == 0:
    print(f"Rank {rank}: Memory allocated: {memory_allocated}")


dist.destroy_process_group()
