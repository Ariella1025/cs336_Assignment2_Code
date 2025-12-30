import math


def lr_Scheduling(t: int, alpha_max: float, alpha_min: float, Tw: float, Tc: float):
    """
    学习率调度器（用于在第t个bacth时调整学习率）
    :param t: 当前迭代次数
    :param alpha_max: 最大学习率
    :param alpha_min: 最小学习率
    :param Tw: 预热轮数
    :param Tc: 最大轮数(在此之后使用最小学习率)
    """
    if t < Tw and t >= 0:
        alpha_t = t/Tw * alpha_max
    if t >= Tw and t <= Tc:
        alpha_t = alpha_min + 0.5 * \
            (1 + math.cos((t-Tw)*math.pi/(Tc-Tw)))*(alpha_max - alpha_min)
    if t > Tc:
        alpha_t = alpha_min
    return alpha_t
