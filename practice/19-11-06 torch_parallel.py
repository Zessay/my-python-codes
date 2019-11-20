'''
关于实现pytorch并行：https://pytorch.org/tutorials/intermediate/dist_tuto.html
'''

import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# 定义自己的模型
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# 配置执行的url和端口
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    ## 通过初始化，确保每个进程都能和主进程协同，使用相同的ip和端口
    ## 这里的本质是，通过进程之间共享各自的位置，实现通信
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 指定seed，保证不同process中的模型初始化参数是相同的
    torch.manual_seed(42)


# 清理所有的进程
def cleanup():
    dist.destroy_process_group()



def demo_basic(rank, world_size):
    setup(rank, world_size)

    # 为进程设定设备，rank1使用[0,1,2,3]GPU，rank2使用[4,5,6,7]GPU
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # 构造模型，并移动到指定的device上
    model = ToyModel().to(device_ids[0])
    # 输出的device默认指定为device[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    ## 运行模型
    outputs = ddp_model(torch.randn(20, 10))
    ## 使用第一个GPU设备进行最终的汇总
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


# 保存checkpoints
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))
    model = ToyModel().to(device_ids[0])
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = "./tmp/model.chekpoint"

    # if rank == 0:
    torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # 使用barrier()函数，当进程0保存模型之后，进程1加载
    dist.barrier()
    # 正确配置map_location，将之前保存模型的GPU映射到当前使用的GPU编号
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
    )

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()

    optimizer.step()

    # 使用barrier确保所有进程读取了checkpoint
    dist.barrier()

    torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    # if rank == 0:
    #     os.remove(CHECKPOINT_PATH)
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size, ), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_demo(demo_checkpoint, 4)