import torch
import torch.nn as nn
s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000): 
    s += torch.tensor(0.01,dtype=torch.float32) 
print(s)

s = torch.tensor(0,dtype=torch.float16) 
for i in range(1000): 
    s += torch.tensor(0.01,dtype=torch.float16) 
print(s)

s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000): 
    s += torch.tensor(0.01,dtype=torch.float16) 
print(s)

s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000): 
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int): 
        super().__init__() 
        self.fc1 = nn.Linear(in_features, 10, bias=False) 
        self.ln = nn.LayerNorm(10) 
        self.fc2 = nn.Linear(10, out_features, bias=False) 
        self.relu = nn.ReLU()  
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("x.dtype after fc1 and relu:", x.dtype)
        x = self.ln(x)
        print("x.dtype after ln:", x.dtype)
        x = self.fc2(x) 
        print("x.dtype after fc2:", x.dtype)
        return x

# 初始化模型
model = ToyModel(in_features=5, out_features=2)
x = torch.randn(3, 5)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with torch.autocast(device_type=device, dtype=torch.float16):
    optimizer.zero_grad()

    output = model(x)
    print(f"模型参数dtype: {next(model.parameters()).dtype}")

    loss = output.sum()
    print("loss dtype:", loss.dtype)
    loss.backward()
    print(f"模型fc1梯度dtype: {model.fc1.weight.grad.dtype}")
    print(f"模型fc2梯度dtype: {model.fc2.weight.grad.dtype}")
    optimizer.step()

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    optimizer.zero_grad()

    output = model(x)
    print(f"模型参数dtype: {next(model.parameters()).dtype}")

    loss = output.sum()
    print("loss dtype:", loss.dtype)
    loss.backward()
    print(f"模型fc1梯度dtype: {model.fc1.weight.grad.dtype}")
    print(f"模型fc2梯度dtype: {model.fc2.weight.grad.dtype}")
    optimizer.step()