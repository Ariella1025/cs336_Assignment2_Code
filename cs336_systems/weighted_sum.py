import triton  # Triton 提供自定义 GPU Kernel 的编译/运行能力（Python 装饰器 + JIT 到 GPU）
import triton.language as tl  # Triton DSL，tl.xx 是编译期/运行期算子（如 tl.load/tl.store/tl.cdiv）
import torch  # PyTorch 主库，负责张量、autograd 与 CUDA 管理
from einops import rearrange  # 重排/展平张量的语法糖；纯 Python 层操作
from torch import Tensor
from jaxtyping import Float  # 类型提示工具，不影响运行，仅用于静态检查


def weighted_sum(x, weight):
    # 纯 PyTorch 参考实现：逐元素乘积后沿最后一维求和
    # axis=-1 指定最后一维，广播规则自动匹配 [..., D] 与 [D]
    return (weight * x).sum(axis=-1)


def _assert_contiguous(x: torch.Tensor) -> torch.Tensor:
    # x.is_contiguous(): 检查内存布局是否默认连续（行主序）
    # 若不连续，则 x.contiguous() 会复制一份连续内存，确保 stride 可预测
    return x if x.is_contiguous() else x.contiguous()

@triton.jit
def weighted_sum_fwd(
    # --- 指针参数 ---
    x_ptr,           # X 矩阵首地址 (形状视为 ROWS x D)
    weight_ptr,      # weight 向量首地址 (形状视为 D)
    output_ptr,      # 输出首地址 (形状视为 ROWS)
    
    # --- 步长参数 (Strides) ---
    # 步长=沿某维前进一步所跳过的元素数
    x_stride_row,    # X 行步长，等于 x.stride(0)
    x_stride_dim,    # X 列步长，等于 x.stride(1)
    weight_stride_dim, # weight 步长，等于 weight.stride(0)，通常为 1
    output_stride_row, # 输出步长，等于 output.stride(0)，通常为 1
    
    # --- 维度参数 ---
    ROWS,            # 行数
    D,               # 列数 / weight 长度
    
    # --- 编译时常量 (constexpr) ---
    ROWS_TILE_SIZE: tl.constexpr, # 编译期常量：每个程序实例覆盖的行数
    D_TILE_SIZE: tl.constexpr,    # 编译期常量：每次迭代加载的列块大小
):
    """
    计算行方向的加权和：output[row] = sum_j X[row, j] * weight[j]
    """
    
    # tl.program_id(0): 取第 0 维 grid 的 program index，决定处理哪一组行
    row_tile_idx = tl.program_id(0)

    # tl.make_block_ptr: 基址 + 逻辑形状 + 步长 + 初始 offsets + block_shape
    # offsets=(row_tile_idx * ROWS_TILE_SIZE, 0) 表示按行块跳过前面的行
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),  # order 决定迭代优先级；(1,0) 表示先列后行的访存顺序
    )

    # weight 为 1D，shape=(D,), stride=(weight_stride_dim,)
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 输出也是 1D，按行块偏移
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 片上累加器：大小为行块，dtype 设为 float32 以累积精度
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # tl.cdiv(D, D_TILE_SIZE): 向上取整的块数，用 for range 在编译期展开
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前行块与权重块，不足处自动填零
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # 广播 weight 后做乘积并在列维度求和
        output += tl.sum(row * weight[None, :], axis=1)

        # 块指针前移到下一列分块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # 将累加结果写回显存
    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,        # 正向输入
    grad_output_ptr,          # 上游梯度
    grad_x_ptr, partial_grad_weight_ptr, # 梯度缓冲区
    stride_xr, stride_xd,     # x 的步长
    stride_wd,                # weight 的步长
    stride_gr,                # grad_output 的步长
    stride_gxr, stride_gxd,   # grad_x 的步长
    stride_gwb, stride_gwd,   # 部分 grad_weight 的步长
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    # 取当前行块索引与总行块数，决定写入 partial grad_weight 的行
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # 上游梯度视为形状 (NUM_ROWS,)
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # x 视为 (NUM_ROWS, D)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # weight 视为 (D,)
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # grad_x 视为 (NUM_ROWS, D)
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # partial_grad_weight 视为 (n_row_tiles, D)，先按行块累加，之后 Python 端再规约
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # 与前向相同的列分块循环
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # grad_x：grad_output 外积 weight
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # grad_weight 部分和：sum_over_rows(x * grad_output)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        # 指针推进到下一列块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

# 定义一个继承自 torch.autograd.Function 的类，用于自定义前向传播和反向传播逻辑
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # 取出特征维 D 和除末维以外的输出形状
        D, output_dims = x.shape[-1], x.shape[:-1]

        # 记录原始形状以便恢复
        input_shape = x.shape
        
        # rearrange("... d -> (...) d"): 保留所有前缀维度为一维 ROWS，最后一维为 d
        x = rearrange(x, "... d -> (...) d")
        
        # 保证 contiguous，避免 stride 异常导致指针算术错误
        x = _assert_contiguous(x)
        weight = _assert_contiguous(weight)

        # 保存到 ctx 供 backward 使用
        ctx.save_for_backward(x, weight)

        # 简单的形状与设备校验
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"

        # triton.next_power_of_2(D)//16: 将 D 补到 2 的幂后再除 16，控制列块大小
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 
        # 行块大小固定为 16 行
        ctx.ROWS_TILE_SIZE = 16 
        ctx.input_shape = input_shape

        # torch.empty 仅分配显存，不初始化；device=x.device 保持同 GPU
        y = torch.empty(output_dims, device=x.device)

        # numel() 返回展平元素数，即行数 ROWS
        n_rows = y.numel()
        
        # 启动前向 Kernel；grid[0]=ceil(n_rows / ROWS_TILE_SIZE)
        weighted_sum_fwd[(tl.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),  # stride(k) 返回第 k 维步长（元素数）
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        # 还原回原批次/序列形状
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        # 取出正向缓存与分块参数
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # partial_grad_weight: 大小 = 行块数 x D；稍后在 Python 端 sum(axis=0)
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        # 启动反向 Kernel；grid 与前向相同，只是写入梯度缓冲
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight, grad_out, grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # 对行块求和得到最终 grad_weight
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x.view(ctx.input_shape), grad_weight