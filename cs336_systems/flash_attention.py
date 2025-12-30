from email.mime import base
from operator import is_
from re import L
from cycler import V
from numpy import block
from regex import D
from sympy import Q
import triton
import triton.language as tl
import torch
# %%

@triton.jit
def flash_attentionV2_fwd(
    Q_ptr, K_ptr, V_ptr,        # 输入Tensor的起始位置
    O_ptr, L_ptr,               # 输出Tensor的起始位置
    stride_qb, stride_qq, stride_qd,        # Q的步长(批次大小，序列长度，特征维度)
    stride_kb, stride_kk, stride_kd,        # K的步长(批次大小，序列长度，特征维度)
    stride_vb, stride_vk, stride_vd,        # V的步长(批次大小，序列长度，特征维度)
    stride_ob, stride_oq, stride_od,        # O的步长(批次大小，序列长度，特征维度)
    stride_lb, stride_lq,                   # L的步长(批次大小，序列长度)
    N_QUERIES, N_KEYS,                      # Q: (bacth_size, N_QUERIES, D), K: (bacth_size, N_KEYS, D)
    scale,                                  # math.sqrt(d_model)
    D: tl.constexpr,                        # 每次迭代加载的特征维度块大小
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,       # 每次迭代加载的Q和K的块大小
    is_causal: tl.constexpr
):
    """
    使用triton实现flash attention前向传播
    Q: (batch_size, N_QUERIES, D)
    K: (batch_size, N_KEYS, D)
    V: (batch_size, N_KEYS, D)

    O: (batch_size, N_QUERIES, D)
    L: (batch_size, N_QUERIES)  # log-sum-exp for each query
    """
    query_tile_idx = tl.program_id(0)  # 按query在N_QUERIES维度划分
    batch_idx = tl.program_id(1)       # 按batch在batch_size维划分

    # 构建Q的block指针
    """
    创建了一个指向Q矩阵中某个块的指针，该块对应于特定的batch和query范围。
    program_id(1)用于选择batch，program_id(0)用于选择query块。
    该指针包含了块的形状、步长和偏移信息，便于在Triton内核中高效访问Q矩阵的数据。
    """
    # 注意每一个实例处理Q中的一个query块, 尺寸为(Q_TILE_SIZE, D)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb,         # 基础地址, 从对应的batch开始
        shape=(N_QUERIES, D),                       # 当前batch的逻辑形状
        strides=(stride_qq, stride_qd),             # 当前batch的步长
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),  # 初始偏移, 定位到当前query块
        block_shape=(Q_TILE_SIZE, D),               # 每个block的形状
        order=(1, 0),                               # 先按D维度访问，再按N_QUERIES维度访问(行优先)
    )

    # 对于K矩阵, 当前实例内的Q_tile则需要和所有的K_tile进行点积计算, 然后再重新组合
    # 阅读文档中fwd的伪代码，每一个Q_tile都需要遍历所有的K_tile和V_tile
    Kt_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),                             
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1)        # K是转置
    )
    # 转置之后, K:(D, N_KEYS), 单位块形状:(D, K_TILE_SIZE), 步长也变

    Vt_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # 输出的O矩阵的block指针
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    # 输出L矩阵的block指针
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # 初始化O和L
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # 初始化当前的log-sum-exp最大值
    max = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    # 加载当前实例的Q_tile, 不足分配补0
    Qt = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    Qt = (Qt * scale).to(tl.float32)   # 缩放

    # 确定需要执行的行数(若有is_causal则可能不需要执行到所有N_KEYS, mask是执行到Key)
    # 如果N_QUERIES<N_KEYS且需要执行casual mask, 则需要mask掉Key的多余部分, 即没必要计算
    # 为什么不是N_QUERIES? 因为一块一块加载Q_tile, 每个tile有Q_TILE_SIZE行, 最后一块会被padding到Q_TILE_SIZE
    N_KEYS_eff = (query_tile_idx+1) * Q_TILE_SIZE if is_causal else N_KEYS       

    # 执行当前实例的计算
    for j in range(tl.cdiv(N_KEYS_eff, K_TILE_SIZE)):
        # 加载当前的K_tile和V_tile
        Kt = tl.load(Kt_block_ptr, boundary_check=(0,1), padding_option="zero")
        Vt = tl.load(Vt_block_ptr, boundary_check=(0,1), padding_option="zero")

        # 计算注意力分数
        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(Qt, Kt, acc=S)       # 累加到S上(scale已经在Qt中处理)
    
        # 执行causal mask
        if is_causal:
            # 当前实例处理的Q_tile在整个Q中的位置, 形状扩展为(Q_TILE_SIZE, 1)
            q_indices = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            # 当前循环处理的K_tile在整个K中的位置, 形状扩展为(1, K_TILE_SIZE)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            mask = q_indices < k_indices    # shape: (Q_TILE_SIZE, K_TILE_SIZE)
            S = tl.where(mask, -1e9, S)     # 执行mask

        # 计算当前块的log-sum-exp
        new_max = tl.maximum(max, tl.max(S, axis=-1))     # rowmax(S_j), 即在最后一个维度上比较
        P = tl.math.exp(S - new_max[:, None])             # 减去new_max以稳定计算, new_max[:, None]广播到每一行

        # 缩放因子
        alpha = tl.math.exp(max - new_max)

        # 更新L和O
        L = alpha * L + tl.sum(P, axis=-1)
        O = alpha[:, None] * O      # 注意这里的广播
        O = tl.dot(P, Vt, acc=O)    # 累加当前块的加权和
        max = new_max

        # 推进KV block ptr
        Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE)) # K已转置
        Vt_block_ptr = Vt_block_ptr.advance((K_TILE_SIZE,0))
    
    O /= L[:, None]       # 归一化O
    L = max + tl.log(L)   # 计算最终的log-sum-exp

    # 将结果写回内存
    tl.store(O_block_ptr, O, boundary_check=(0,1))
    tl.store(L_block_ptr, L, boundary_check=(0,))
# %%
@triton.jit
def flash_attentionV2_bwd_q(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr,
    dQ_ptr, 
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dqb, stride_dqq, stride_dqd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    """
    使用triton实现flash attention反向传播中dQ的计算
    Q: (batch_size, N_QUERIES, D)
    K: (batch_size, N_KEYS, D)
    V: (batch_size, N_KEYS, D)
    dO: (batch_size, N_QUERIES, D)
    L: (batch_size, N_QUERIES)
    D: (batch_size, N_QUERIES)  # 中间变量
    dQ: (batch_size, N_QUERIES, D)
    """

    query_tile_idx = tl.program_id(0)  # 按query在N_QUERIES维度划分
    batch_idx = tl.program_id(1)       # 按batch在batch_size维划分

    # 创建输入输出的block指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_idx * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    # 输出
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_idx * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # 加载当前实例的Q_tile, dO_tile, L_tile, D_tile
    Qt = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    dOt = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    Lt = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Dt = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # 初始化dQ
    dQt = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # 遍历所有的K_tile和V_tile
    N_KEYS_eff = (query_tile_idx+1) * Q_TILE_SIZE if is_causal else N_KEYS
    for j in range(tl.cdiv(N_KEYS_eff, K_TILE_SIZE)):
        Kt = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        Vt = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

        # 计算注意力分数
        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(Qt * scale, tl.trans(Kt), acc=S).to(tl.float32)

        # 执行causal mask
        if is_causal:
            q_indices = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            mask = q_indices < k_indices
            S = tl.where(mask, -1e9, S)

        # 计算P矩阵
        P = tl.math.exp(S - Lt[:, None])

        dP = tl.dot(dOt, tl.trans(Vt)).to(tl.float32)    # dO:(Q_TILE_SIZE, D), Vt:(K_TILE_SIZE, D)

        dS = P * (dP - Dt[:, None]) * scale    # 计算dS
        if is_causal:
            dS = tl.where(mask, 0.0, dS)

        # 计算并累加dQ
        dQt = tl.dot(dS, Kt, acc=dQt).to(tl.float32)

        # 推进指针
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # 写回dQ
    tl.store(dQ_block_ptr, dQt, boundary_check=(0,1))


@triton.jit
def flash_attentionV2_bwd_kv(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    """
    使用triton实现flash attention反向传播中dK和dV的计算
    Q: (batch_size, N_QUERIES, D)
    K: (batch_size, N_KEYS, D)
    V: (batch_size, N_KEYS, D)
    dO: (batch_size, N_QUERIES, D)
    L: (batch_size, N_QUERIES)
    D: (batch_size, N_QUERIES)  # 中间变量
    dK: (batch_size, N_KEYS, D)
    dV: (batch_size, N_KEYS, D)
    """
    key_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    K_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_kb + K_ptr,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (key_tile_idx * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_vb + V_ptr,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (key_tile_idx * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )

    Qt_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_qb + Q_ptr,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )

    dOt_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_qb + dO_ptr,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )

    Lt_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_lb + L_ptr,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,)
    )

    Dt_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_db + D_ptr,
        shape = (N_QUERIES,),
        strides = (stride_dq,),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,)
    )
    dK_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_dkb + dK_ptr,
        shape = (N_KEYS, D),
        strides = (stride_dkk, stride_dkd),
        offsets = (key_tile_idx * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        base = batch_idx * stride_dvb + dV_ptr,
        shape = (N_KEYS, D),
        strides = (stride_dvk, stride_dvd),
        offsets = (key_tile_idx * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )

    # 加载当前实例的K_tile和V_tile
    Kt = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
    Vt = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

    # 初始化dK和dV
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # 遍历所有的Q_tile
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Qt = tl.load(Qt_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        dOt = tl.load(dOt_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        Lt = tl.load(Lt_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Dt = tl.load(Dt_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # 计算注意力分数
        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(Qt * scale, tl.trans(Kt)).to(tl.float32)

        # 执行causal mask
        if is_causal:
            q_indices = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_indices = key_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            mask = q_indices < k_indices
            S = tl.where(mask, -1e9, S)

        P = tl.math.exp(S - Lt[:, None])    # shape: (Q_TILE_SIZE, K_TILE_SIZE)
        dV = tl.dot(tl.trans(P), dOt, acc=dV)
        dP = tl.dot(dOt, tl.trans(Vt))                             # (Q_TILE_SIZE, K_TILE_SIZE)
        dS = (dP - Dt[:, None]) * P * scale                        # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            dS = tl.where(mask, 0.0, dS)
            
        dK = tl.dot(tl.trans(dS), Qt, acc=dK)

        # 推进Q, dO, L, D的block指针
        Qt_block_ptr = Qt_block_ptr.advance((Q_TILE_SIZE, 0))
        dOt_block_ptr = dOt_block_ptr.advance((Q_TILE_SIZE, 0))
        Lt_block_ptr = Lt_block_ptr.advance((Q_TILE_SIZE,))
        Dt_block_ptr = Dt_block_ptr.advance((Q_TILE_SIZE,))
    
    # 写回dK和dV
    tl.store(dK_block_ptr, dK, boundary_check=(0,1))
    tl.store(dV_block_ptr, dV, boundary_check=(0,1))

class FlashAttentionV2_Triton(torch.autograd.Function):
    """
    Flash Attention V2 implementation using Triton.
    """
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_casual:bool=False, mask:torch.Tensor | None=None):
        """
        Forward pass for Flash Attention V2.

        Args:
            Q: Query tensor of shape (batch_size, N_QUERIES, D)
            K: Key tensor of shape (batch_size, N_KEYS, D)
            V: Value tensor of shape (batch_size, N_KEYS, D)
            is_causal: Whether to apply causal masking.

        Returns:
            O: Output tensor of shape (batch_size, N_QUERIES, D)
        """
        # 获取相关参数
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        scale = 1.0 / (D ** 0.5)

        # 确认张量在cuda上并且内存连续, 
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Input tensors must be on CUDA device."
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous"

        # 保存在ctx中以便backward使用
        ctx.save_for_backward(Q, K, V)
        ctx.is_casual = is_casual

        # 形状校验
        assert K.shape == (batch_size, N_KEYS, D), "K shape mismatch"
        assert V.shape == (batch_size, N_KEYS, D), "V shape mismatch"

        # 控制Q和KV的行块大小
        ctx.Q_TILE_SIZE = 16            # 至少16行
        ctx.K_TILE_SIZE = 16

        # 分配输出显存
        O = torch.empty((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=Q.dtype)

        # 启动Kernel
        flash_attentionV2_fwd[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch_size, )](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            O_ptr=O, L_ptr=L,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_casual
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """
        Backward pass for Flash Attention V2.

        Args:
            dO: Gradient of the output tensor of shape (batch_size, N_QUERIES, D)

        returns:
            dQ: Gradient of the query tensor of shape (batch_size, N_QUERIES, D)
            dK: Gradient of the key tensor of shape (batch_size, N_KEYS, D)
            dV: Gradient of the value tensor of shape (batch_size, N_KEYS, D)
        """
        # 获取保存的张量
        Q, K, V, O, L = ctx.saved_tensors
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        is_causal = ctx.is_casual
        scale = 1.0 / (D ** 0.5)

        # 计算得到D张量
        D_tensor = (O * dO).sum(dim=-1).to(torch.float32)

        # 确认相关张量连续及设备正确
        assert dO.is_contiguous(), "dO must be contiguous"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous"
        assert D_tensor.is_contiguous(), "D_tensor must be contiguous"
        assert L.is_contiguous(), "L must be contiguous"
        assert dO.is_cuda and Q.is_cuda and K.is_cuda and V.is_cuda and D_tensor.is_cuda and L.is_cuda, "All tensors must be on CUDA device."

        # 控制Q和KV的行块大小
        ctx.Q_TILE_SIZE = 16            # 至少16行
        ctx.K_TILE_SIZE = 16

        # 分配输出显存
        # dQ is accumulated via atomic_add in kernel; must be zero-initialized
        dQ = torch.zeros((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        dK = torch.empty((batch_size, N_KEYS, D), device=K.device, dtype=K.dtype)
        dV = torch.empty((batch_size, N_KEYS, D), device=V.device, dtype=V.dtype)

        # 启动kernel
        if is_causal:
            N_KEYS_eff = N_QUERIES
        else:
            N_KEYS_eff = N_KEYS

        # 计算dQ
        flash_attentionV2_bwd_q[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch_size, )](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            dO_ptr=dO, L_ptr=L,
            dQ_ptr=dQ,
            D_ptr=D_tensor,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_dqb=dQ.stride(0), stride_dqq=dQ.stride(1), stride_dqd=dQ.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            stride_db=D_tensor.stride(0), stride_dq=D_tensor.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal
        )
        
        # 计算dK和dV
        flash_attentionV2_bwd_kv[(triton.cdiv(N_KEYS_eff, ctx.K_TILE_SIZE), batch_size, )](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            dO_ptr=dO, L_ptr=L,
            dK_ptr=dK, dV_ptr=dV,
            D_ptr=D_tensor,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_dkb=dK.stride(0), stride_dkk=dK.stride(1), stride_dkd=dK.stride(2),
            stride_dvb=dV.stride(0), stride_dvk=dV.stride(1), stride_dvd=dV.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            stride_db=D_tensor.stride(0), stride_dq=D_tensor.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal
        )

        return dQ, dK, dV, None

# %%