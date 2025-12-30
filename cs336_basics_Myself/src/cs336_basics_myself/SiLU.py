from turtle import forward
import torch
import torch.nn as nn
import math

def SiLU(x):
    return x*torch.sigmoid(x)