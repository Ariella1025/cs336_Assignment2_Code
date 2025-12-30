import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        构造 RoPE 模块并在需要时创建缓冲区。

        theta: float RoPE 的Theta值
        d_k: int 查询和键向量的维度
        max_seq_len: int 将输入的最大序列长度
        device: torch.device | None = None 存储缓冲区的设备
        注意用复数乘法来代替
        """
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算旋转矩阵(一张大表格, shape为(max_seq_len, d_k))
        # 注意是从0开始, 第m个token(m = 0,1,2,...,max_seq_len - 1)
        position = torch.arange(max_seq_len, device=device).unsqueeze(-1)

        # 频率从 i=0 开始，使用 2i/d_k 指数，与经典RoPE定义一致, 注意不是和文档一致
        i = torch.arange(0, d_k // 2, device=device).unsqueeze(0)
        inv_freq = torch.pow(torch.tensor(
            theta, device=device), -(2 * i) / d_k)

        # 相位 position * inv_freq
        phase = torch.einsum('ab,bc->ac', position, inv_freq)
        # exp(1i*phase)
        Rotary_matrix = torch.exp(1j * phase)
        # 存储旋转矩阵(非参数, 不优化)
        self.register_buffer("Rotary_Matrix", Rotary_matrix,
                             persistent=False)

    @staticmethod
    def to_complex(x: torch.Tensor) -> torch.Tensor:
        """
        把最后一维的 (x1, x2) → (x1 + i*x2)
        输入:  (..., d_k)
        输出:  (..., d_k/2) 复数类型
        """
        x_even = x[..., ::2].float()        # even
        x_odd = x[..., 1::2].float()        # odd
        return torch.complex(x_even, x_odd)

    @staticmethod
    def to_real(x_complex: torch.Tensor) -> torch.Tensor:
        """
        把复数 (a + i*b) 还原成实数拼接 (a, b)
        输入:  (..., d_k/2) 复数类型
        输出:  (..., d_k) 实数类型
        """
        real = x_complex.real
        imag = x_complex.imag
        result = torch.stack([real, imag], dim=-
                             1).reshape(*x_complex.shape[:-1], -1)
        return result

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        处理形状为(..., seq_len, d_k)的输入张量
        并返回相同形状的张量。
        :param x: 输入矩阵, shape一般为(batch_size, seq_len, d_k)
        :param token_positions: 上述输入矩阵中的各个token在原max_seq_len中的位置（长文本推理可能用到）
        例如: 对于batch_size = 1的最大长度为8的序列, 其对应位置id为[0,1,2,3,4,5,6,7], 嵌入维度暂时忽略
        预计算了一张长度为8的旋转矩阵
        对于某些长文本, seq_len太长会截断, 例如当前先输入位置id(token_positions)为[0,1,2]的序列, 此时seq_len = 3
        再输入token_positions为[3,4,5]的序列, 此时seq_len = 3
        旋转矩阵要从buffer中截断出来
        """
        x = x.to(self.device)
        # 查表获取当前输入的旋转矩阵
        # token_positions(1,1) = 0, 获取Rotary_matrix[0,:]填入token_positions(1,1)
        # 得到旋转矩阵(batch_size, seq_len, d_k/2)
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)
        token_positions = token_positions.to(self.device)
        rotary_matrix = self.Rotary_Matrix[token_positions]

        # 重构输入x矩阵, 准备进行复数乘法
        # 拆最后一个维度的向量, 例如(x1, x2)进行复数化为x1+i*x2, 张量维度为(batch_size, seq_len, d_k/2)
        x_complex = self.to_complex(x)

        # 执行复数乘法（逐元素）
        result_complex = rotary_matrix * x_complex
        return self.to_real(result_complex)
