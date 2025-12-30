from re import L
import torch
import math

def softmax(x: torch.Tensor, dim: int):
    """在指定维度上计算softmax"""
    numerator = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

def flash_attentionV2_fwd(
        Q: torch.Tensor, 
        K:torch.Tensor, 
        V:torch.Tensor, 
        is_casual:bool=False, 
        mask:torch.Tensor | None=None):
    """纯Pytorch实现flash attention前向传播"""
    d_model = Q.shape[-1]
    # scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    S = torch.einsum("...id, ...jd ->...ij", Q, K) / math.sqrt(d_model)

    # 应用mask和causal掩码
    if mask is not None and is_casual:
        S = S.masked_fill(~mask, -torch.inf)

    # logsumexp技巧计算logsumexp
    L = torch.log(torch.sum(torch.exp(S), dim=-1))

    # 计算输出
    O = softmax(S, dim=-1) @ V
    return O, L

class FlashAttentionV2_Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_casual:bool=False, mask:torch.Tensor | None=None):
        """
            Q: (..., N_QUERIES, D)
            K: (..., N_KEYS, D)
            V: (..., N_KEYS, D)
            O: (..., N_QUERIES, D)
            L: (..., N_QUERIES)
        """
        if mask is not None:
            ctx.mask = mask
        O, L = flash_attentionV2_fwd(Q, K, V, is_casual, mask)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_casual = is_casual
        return O
        

    @staticmethod
    @torch.compile(fullgraph=True)
    def backward(ctx, O_grad):
        Q, K, V, O, L = ctx.saved_tensors
        is_casual = ctx.is_casual

        d_model = Q.shape[-1]
        S = torch.einsum("...id, ...jd ->...ij", Q, K) / math.sqrt(d_model)

        if is_casual:
            mask = torch.tril(torch.ones(S.shape[-2], S.shape[-1], device=S.device)).bool()
            S = S.masked_fill(~mask, -torch.inf)

        P = torch.exp(S - L.unsqueeze(-1))  # softmax概率矩阵

        V_grad = torch.einsum("...ij, ...id ->...jd", P, O_grad)
        P_grad = torch.einsum("...id, ...jd ->...ij", O_grad, V)

        D = (O * O_grad).sum(dim=-1)
        S_grad = P * (P_grad - D.unsqueeze(-1))

        Q_grad = torch.einsum("...ij, ...jd ->...id", S_grad, K) / math.sqrt(d_model)

        K_grad = torch.einsum("...ij, ...id ->...jd", S_grad, Q) / math.sqrt(d_model)

        return Q_grad, K_grad, V_grad, None