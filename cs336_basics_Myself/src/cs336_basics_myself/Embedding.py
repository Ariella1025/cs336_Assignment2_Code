import math
import torch
from torch.mtia import device
import torch.nn as nn


class Embedding(nn.Module):
    """将整数token id映射到维度指定的向量空间"""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        构造一个嵌入模块。此函数应接受以下参数：
        :param num_embeddings: int 词汇表的大小(vocab_size)
        :param embedding_dim: int 嵌入向量的维度，即需要映射到的维度(d_model)
        :param device: torch.device | None = None 存储参数的设备
        :param dtype: torch.dtype | None = None 参数的数据类型
        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # 嵌入矩阵(num_embeddings(vocab_size), embedding_dim(d_model))
        self.embedding = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype))

        # 初始化
        self.reset_parameter()

    # 初始化
    def reset_parameter(self):
        """参数初始化"""
        # 原始初始化（保留参考）
        mean = 0
        std = math.sqrt(1.0 / max(1, self.embedding_dim))
        a = -3 * std
        b = 3 * std
        nn.init.trunc_normal_(self.embedding, mean=mean, std=std, a=a, b=b)

        # # 当前初始化：较小方差的截断正态，降低早期激活尺度
        # mean = 0.0
        # std = 0.002
        # a = -2 * std
        # b = 2 * std
        # nn.init.trunc_normal_(self.embedding, mean=mean, std=std, a=a, b=b)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        查找给定 token ID 的嵌入向量。
        :param token_ids: 输入token的id(batch_size, seq_len)
        """
        # 将输入的token按照num_embeddings(vocab_size)转为one-hot编码
        # token_ids_onehot(batch_size, seq_len, num_embeddings(vocab_size))
        token_ids = token_ids.to(self.device)

        # 直接用索引查找 embedding（避免创建 one-hot 张量）
        # self.embedding 的 shape: (vocab_size, embedding_dim)
        output = self.embedding[token_ids]
        return output
