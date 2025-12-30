import numpy as np
import torch
import torch.nn as nn
import numpy.typing as npt
from typing import Iterator


class DataLoader:
    """数据下载"""

    def __init__(self, batch_size: int, context_length: int, device: str = 'cpu'):
        """
        初始化一个数据下载器
        :param batch_size: 单batch样本数
        :param context_length: 单样本上下文长度
        :param device: 数据所在的设备("cpu"或着"cuda: 0")
        """
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_train_batch_data(self, dataset: npt.NDArray) -> tuple[torch.Tensor, torch.Tensor]:
        """获得一个batch的训练数据(batch_size, context_length)"""
        N = len(dataset)

        # 随机产生起点, 一共batch_size个
        # 超大数据可以容忍重复样本
        starts = np.random.randint(
            0, N-self.context_length, size=self.batch_size)

        # 输入序列
        x = np.stack([dataset[s: s + self.context_length] for s in starts])
        # 输出序列
        y = np.stack([dataset[s+1: s+1 + self.context_length] for s in starts])

        # 转tensor并放在规定的设备上
        x = torch.from_numpy(x).long().to(self.device)
        y = torch.from_numpy(y).long().to(self.device)

        return x, y
    
    # 原有错误版本
    # def get_valid_batch_data_iter(self, dataset: npt.NDArray) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    #     """获得验证集产生的非重叠数据生成器"""
    #     # 移除末尾不完整的元素，确保整个数据集能被 context_length * batch_size 整除
    #     total_len = len(dataset)
    #     num_full_elements = total_len - \
    #         (total_len % (self.context_length * self.batch_size))

    #     # 将整个数据集切分成 batch_size 个不重叠的“子流”
    #     # 然后将每个子流重塑成 (num_sequences, context_length) 的形状
    #     data = dataset[:num_full_elements].reshape(self.batch_size, -1)

    #     # 计算每个子流中的序列数量
    #     num_sequences = data.shape[1] // self.context_length

    #     # 遍历所有序列
    #     for i in range(num_sequences - 1):  # 留出最后一个序列用于y
    #         x = data[:, i * self.context_length: (i+1) * self.context_length]
    #         y = data[:, (i+1) * self.context_length: (i+2)
    #                  * self.context_length]

    #         x_tensor = torch.from_numpy(x.copy()).long().to(self.device)
    #         y_tensor = torch.from_numpy(y.copy()).long().to(self.device)

    #         yield x_tensor, y_tensor

    def get_valid_batch_data_iter(self, dataset: npt.NDArray, valid_batches = None):
        total_len = len(dataset)
        

        # 先裁剪成 batch_size * seq_len
        usable_len = total_len - (total_len % self.batch_size)
        dataset = dataset[:usable_len].reshape(self.batch_size, -1)
        seq_len = dataset.shape[1]

        # 提取满足 seq_len = self.context_length*k + 1 的最大子序列长度
        usable_seq_len = ((seq_len - 1) // self.context_length) * self.context_length + 1
        dataset = dataset[:, :usable_seq_len]

        for i in range(0, usable_seq_len - 1, self.context_length):
            x = dataset[:, i: i + self.context_length]
            y = dataset[:, i + 1: i + 1 + self.context_length]

            x_tensor = torch.from_numpy(x.copy()).long().to(self.device)
            y_tensor = torch.from_numpy(y.copy()).long().to(self.device)

            if valid_batches is not None:
                if i // self.context_length >= valid_batches:
                    break

            yield x_tensor, y_tensor