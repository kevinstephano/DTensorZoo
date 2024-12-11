import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, distribute_tensor, init_device_mesh

"""
To run the example on 8 gpus, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=8 dtest_simple.py
"""

# construct a device mesh with available devices (multi-host or single host)
device_mesh = init_device_mesh("cuda", (8,))
rank = dist.get_rank()

# if we want to do col-wise sharding
rowwise_placement=[Shard(1)]

big_tensor = torch.empty(888, 16) # CPU Tensor
if rank == 0:
    print("Global Tensor Size:", big_tensor.size(), "Stride:", big_tensor.stride())
# distributed tensor returned will be sharded across the dimension specified in placements
rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)

if rank == 0:
    print("Local Tensor Device", rowwise_tensor._local_tensor.device, "Size:", rowwise_tensor._local_tensor.size(), "Stride:", rowwise_tensor._local_tensor.stride())

# Check memory consumption
memory_allocated = torch.cuda.memory_allocated()
if rank == 0:
    print(f"Rank {rank}: Memory allocated: {memory_allocated}")

dist.destroy_process_group()
