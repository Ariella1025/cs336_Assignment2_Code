import torch
import torch.nn as nn
from .Linear import Linear
from .Embedding import Embedding
from .RMSNorm import RMSNorm
from .Pre_Transformer import PreTransformer

class TransformerLM(nn.Module):
    """Transformer模型"""

    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 device=None
                 ):
        """
        实现一个完整的 Pre-Norm Transformer 模块
        :param vocab_size: int,词表大小
        :param context_length: int 最大上下文长度（序列长度）, 决定位置编码的范围。
        :param num_layers: int TransformerBlock 堆叠的层数。
        :param d_model: int Transformer 块输入与输出的维度（即嵌入维度）。
        :param num_heads: int 在多头自注意力中使用的注意力头数。
        :param d_ff: int 前馈网络隐藏层的维度。
        :param theta: 旋转编码使用的theta
        """
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device

        # 初始化模块
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.TransformerBlock = nn.ModuleList([
            PreTransformer(d_model, num_heads, d_ff, device=device)
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # x.shape:(bacth_size, seq_len)
        x = x.to(self.device)
        x_emb = self.embedding(x)
        # x_emb.shape: (batch_size, seq_len, d_model)

        x_input = x_emb
        for block in self.TransformerBlock:
            x_attn_output = block(
                x_input, (self.theta, self.context_length, token_positions))
            x_input = x_attn_output
        # x_attn_output.shape: (batch_size, seq_len, d_model)

        x_norm = self.rmsnorm(x_attn_output)
        # x_norm.shape: (batch_size, seq_len, d_model)

        result = self.linear(x_norm)
        # result.shape: (batch_size, seq_len, vocab_size)

        return result
