import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

"""
To run the example, use the following command:
TORCH_COMPILE_DEBUG=1 torchrun --standalone --nnodes=1 --nproc-per-node=8 dtest_linear_rowwise.py
"""

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.relu(out1)
        return out2

mesh = init_device_mesh("cuda", (8,))
rank = dist.get_rank()

def shard_params(mod_name, mod, mesh):
    row_linear_placement = [Shard(1)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, row_linear_placement)
            )
            mod.register_parameter(name, dist_param)

model = distribute_module(MyModule(), mesh, partition_fn=shard_params)

inputs = torch.randn(16, 512, 4096, device='cuda')
dinputs = distribute_tensor(inputs, mesh, [Replicate()])

model = torch.compile(model)

out = model(dinputs)

# Check memory consumption
memory_allocated = torch.cuda.memory_allocated()
if rank == 0:
    print(f"Rank {rank}: Memory allocated: {memory_allocated}")

dist.destroy_process_group()
