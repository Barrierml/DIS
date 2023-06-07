import torch
import torch.nn as nn
import numpy as np

# 假设你有两个图像预测处理模型 model1 和 model2
# 假设他们的输出都是 H x W x C 的张量
# 假设你想要在后面训练一个全连接层 fc_layer

# 定义一个特征融合的类，可以选择不同的方法来融合两个模型的输出
class FeatureFusion(nn.Module):
  def __init__(self, method="concat", C = 1):
    super(FeatureFusion, self).__init__()
    assert method in ["concat", "sum", "mul", "att", "pyr"]
    self.method = method
    if self.method == "att":
      # 如果使用注意力机制，需要定义一个注意力权重矩阵
      self.att_weight = nn.Parameter(torch.randn(C, C))
    if self.method == "pyr":
      # 如果使用金字塔池化，需要定义不同尺度的池化层
      self.pyr_pooling = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(i, i)) for i in [1, 2, 3, 6]])

  def forward(self, x1, x2):
    # x1 和 x2 是两个模型的输出，形状都是 B x C x H x W
    if self.method == "concat":
      # 拼接方法，直接在通道维度上拼接，输出形状为 B x 2C x H x W
      output = torch.cat((x1, x2), dim=1)
    elif self.method == "sum":
      # 相加方法，直接逐元素相加，输出形状为 B x C x H x W
      output = x1 + x2
    elif self.method == "mul":
      # 相乘方法，直接逐元素相乘，输出形状为 B x C x H x W
      output = x1 * x2
    elif self.method == "att":
      # 注意力方法，先计算注意力分数，然后对 x2 加权求和，输出形状为 B x C x H x W
      att_score = torch.matmul(x1.permute(0, 2, 3, 1), self.att_weight) # B x H x W x C
      att_score = torch.matmul(att_score, x2.permute(0, 2, 3, 1).unsqueeze(-1)) # B x H x W x 1
      att_score = torch.softmax(att_score, dim=1) # 对每个像素位置进行 softmax 归一化
      output = att_score * x2 # B x C x H x W
    elif self.method == "pyr":
      # 金字塔池化方法，先对两个输出求和，然后进行不同尺度的池化，最后拼接起来，输出形状为 B x (C + 4C) x H x W
      output = []
      output.append(x1 + x2) # B x C x H x W
      for pool in self.pyr_pooling:
        pooled = pool(x1 + x2) # B x C x i x i
        upsampled = nn.functional.interpolate(pooled, size=(H, W), mode="bilinear") # B x C X H X W
        output.append(upsampled)
      output = torch.cat(output, dim=1) # B X (C + 4C) X H X W
    
    return output

# 定义一个全连接层的类，可以根据输入和输出的大小自动调整参数
class FullyConnectedLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(FullyConnectedLayer, self).__init__()
    self.fc_layer = nn.Linear(input_size, 10)
    self.fc_layer1 = nn.Linear(10, 10)
    self.fc_layer2 = nn.Linear(10, 10)
    self.fc_layer3 = nn.Linear(10, output_size)
    # 激活函数为 relu
    self.act_fn1 = nn.Sigmoid()
    self.act_fn2 = nn.ReLU()
    self.act_fn3 = nn.ReLU()

  def forward(self, input):
    # input 的形状为 B X C X H X W
    input = input.view(input.size(0), -1) # 展平成一维向
    res = self.fc_layer(input) # B X 10
    res = self.act_fn1(res)
    res = self.fc_layer1(res) # B X 10
    res = self.act_fn2(res)
    res = self.fc_layer2(res) # B X 1
    res = self.act_fn3(res)
    res = self.fc_layer3(res) # B X 1
    return res
  

# 定义一个新的数据加载器，用于全链接层的训练
class NewDataLoader():
  def __init__(self) -> None:
    super(NewDataLoader, self).__init__()
    # 生成 1000 个 -10 - 10 的随机数
    self.allData = np.random.rand(10000) * 20 - 10
  def __iter__(self):
    # 随机选择一个数作为输入
    input = np.random.choice(self.allData, 1)
    output = input * input
    input = torch.from_numpy(input).float()
    output = torch.from_numpy(output).float()
    return iter([(input, output)])

# 测试全链接层
fc_layer = FullyConnectedLayer(1, 1)
dataloader = NewDataLoader()
# 定义损失函数
loss_fn = nn.MSELoss()
# 定义优化器
optimizer = torch.optim.SGD(fc_layer.parameters(), lr=1e-4)

losses = []

# 开始训练
for i in range(100000):
  for input, output in dataloader:
    # 前向传播
    res = fc_layer(input)
    # 计算损失
    loss = loss_fn(res, output)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 1000 == 0:
      print('第 {} 次训练，损失为 {}'.format(i, loss.item()))
  # 如果连续 100 epoch 都没有损失下降的话
  # if i > 1000 and (losses[-2] - losses[-1] < 1e-10):
    # print('没有提升啦，结束结束')
    # break

# 绘制损失曲线
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


# 测试训练好的全连接层，并绘制对应的曲线
import matplotlib.pyplot as plt
# 重新生成测试数据集，并绘制 x ^ 2 的曲线 与 全连接层的输出曲线
# 从 0 -100
test_input = np.arange(-10, 10, 0.1)
test_input = torch.from_numpy(test_input).float()
test_input = test_input.unsqueeze(1)

test_output = []
with torch.no_grad():
  for input in test_input:
    res = fc_layer(input)
    test_output.append(res.item())

# 绘制两个曲线
plt.plot(test_input.view(-1).numpy(), test_output)
plt.plot(test_input.view(-1).numpy(), test_input.view(-1).numpy() * test_input.view(-1).numpy())
plt.show()


print(fc_layer(torch.tensor([11.0])))