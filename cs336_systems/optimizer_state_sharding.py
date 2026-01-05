import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Any, Type

class optimizer_state_sharding(torch.optim.Optimizer):
    """实现一个优化器状态分片的优化器包装类, 用于包装优化器，使其支持优化器状态的分片和通信重叠"""
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        """初始化优化器状态分片包装类"""
        # 当前设备编号
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        # 总设备数量
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # 全量参数备份 + 展平顺序 + 分片参数缓存，在父类初始化前先建好，防止父类触发 add_param_group 时未定义
        self.all_params: list[list[torch.nn.Parameter]] = []      # 按 param group 保存
        self.flat_params: list[torch.nn.Parameter] = []           # 展平后的全局顺序，用于 owner 判定
        self.sharded_params: list[torch.nn.Parameter] = []

        # 通过初始化父类, 调用了重写后的 add_param_group 方法, 保存了全量参数
        # 并执行了参数分片, 同时产生了只保留当前 rank 负责参数的 param_groups
        super().__init__(params, {})

        # 实例化基础优化器
        self.base_optimizer = optimizer_cls(self.param_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """向优化器添加一个参数组, 并进行参数分片, 只保留当前rank负责的参数
            重写了父类的add_param_group方法
        """
        # 备份全量参数
        all_params = param_group['params']  # list of all parameters
        self.all_params.append(all_params)
        self.flat_params.extend(all_params)

        # 按顺序取模并进行参数分片
        # 假设所有 rank 的参数顺序一致, i%world_size 决定哪个 rank 负责该参数的更新
        sharded_params = [p for i, p in enumerate(all_params) 
                          if i % self.world_size == self.rank]
        
        # 存储分片后的参数（累积缓存方便调试/检查）
        self.sharded_params.extend(sharded_params)

        # 更新参数组为分片后的参数; 复制 dict 防止外部引用被修改
        param_group = dict(param_group)
        param_group['params'] = sharded_params

        # 注意, 调用的是原来的add_param_group方法, 而不是当前重写后的递归调用, 将当前rank负责的参数加入优化器
        # 刷新self.param_groups["param"]为分片后的参数
        # 其他的例如lr等超参数保持不变, 全量参数仍然保留
        super().add_param_group(param_group)

    def step(self, closure=None, **kwargs):
        """执行优化器的单步更新, 并进行分片状态的通信和更新"""
        # 更新当前rank负责的参数
        loss = self.base_optimizer.step(closure, **kwargs)

        # 将更新后的参数状态广播给所有rank
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            with torch.no_grad():
                for idx, p in enumerate(self.flat_params):
                    # 依据全局参数顺序判定拥有者，避免裁剪后的 param_groups 索引错位
                    owner_rank = idx % self.world_size
                    # 将拥有者更新后的权重同步到所有rank
                    dist.broadcast(p, src=owner_rank)
        
        return loss

