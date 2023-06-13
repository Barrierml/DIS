import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
from ResUnet import resnet34
import time

# 定义数据加载类
class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, depth_dir, gt_dir, depth_transform=None, gt_transform=None):
        self.depth_dir = depth_dir
        self.gt_dir = gt_dir
        self.depth_transform = depth_transform
        self.gt_transform = gt_transform
        self.filenames = os.listdir(depth_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        depth = Image.open(os.path.join(self.depth_dir, self.filenames[idx]))
        gt = Image.open(os.path.join(self.gt_dir, self.filenames[idx].replace('.jpg', '.png')))
        depth = depth.convert('RGB')
        if self.depth_transform:
            depth = self.depth_transform(depth)
        if self.gt_transform:
            gt = self.gt_transform(gt)
        return depth, gt


# 定义一个计算深度图的损失函数
class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        mse_loss = self.mse_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        return (mse_loss + l1_loss) / 2

# 定义一个计算图像分割的损失函数
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce_loss(pred, target)

# 定义一个计算总损失的函数
def total_loss(pred, target, depth_criterion, seg_criterion, depth_weight=0.5):
    # print(pred, target)
    depth_loss = depth_criterion(pred, target)
    seg_loss = seg_criterion(pred, target)
    return depth_loss * depth_weight + seg_loss * (1 - depth_weight)

# 定义一个计算深度图的准确率的函数
def depth_acc(pred, target):
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return np.sum(pred == target) / target.size

# 定义优化器
def get_optimizer(model, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

# 定义学习率衰减器
def get_lr_scheduler(optimizer, step_size, gamma):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return lr_scheduler


# 定义训练函数
# 深度图处理
depth_transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])
# 图像分割处理
gt_transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

base_dir = '/root/data/'

# 定义训练集
train_dataset = DepthDataset(depth_dir=base_dir + 'meitu/depth', gt_dir=base_dir + 'meitu/gt', depth_transform=depth_transform, gt_transform=gt_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定义验证集
val_dataset = DepthDataset(depth_dir=base_dir + 'test/depth', gt_dir=base_dir + 'test/gt', depth_transform=depth_transform, gt_transform=gt_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

device = 'cuda'

# 定义模型
model = resnet34(3, 1, pretrain=True)
model = model.to(device)
model.train()

print('model loaded')

# 定义损失函数
depth_criterion = DepthLoss()
seg_criterion = SegLoss()


# 定义优化器
optimizer = get_optimizer(model, lr=0.001, weight_decay=0.0001)

# 定义学习率衰减器
lr_scheduler = get_lr_scheduler(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(model, train_loader, optimizer, depth_criterion, seg_criterion, epoch):
    train_loss = 0
    train_depth_acc = 0
    start_time = time.time()
    for i, (depth, gt) in enumerate(train_loader):
        depth = depth.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()
        pred = model(depth)
        loss = total_loss(pred, gt, depth_criterion, seg_criterion)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_depth_acc += depth_acc(pred, gt)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDepth Acc: {:.6f} \t Cost time: {:.6f}'.format(
                epoch, i * len(depth), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item(), depth_acc(pred, gt), time.time() - start_time))
        start_time = time.time()
    train_loss /= len(train_loader)
    train_depth_acc /= len(train_loader)
    print('Train Epoch: {}\tAverage Loss: {:.6f}\tAverage Depth Acc: {:.6f}'.format(
        epoch, train_loss, train_depth_acc))
    return train_loss, train_depth_acc

# 定义验证函数
def val(model, val_loader, depth_criterion, seg_criterion):
    model.eval()
    val_loss = 0
    val_depth_acc = 0
    with torch.no_grad():
        for depth, gt in val_loader:
            depth = depth.to(device)
            gt = gt.to(device)
            pred = model(depth)
            loss = total_loss(pred, gt, depth_criterion, seg_criterion)
            val_loss += loss.item()
            val_depth_acc += depth_acc(pred, gt)
    val_loss /= len(val_loader)
    val_depth_acc /= len(val_loader)
    print('Val: Average Loss: {:.6f}\tAverage Depth Acc: {:.6f}'.format(
        val_loss, val_depth_acc))
    return val_loss, val_depth_acc

# 开始训练
train_losses = []
train_depth_accs = []
val_losses = []
val_depth_accs = []
for epoch in range(1, 21):
    train_loss, train_depth_acc = train(model, train_loader, optimizer, depth_criterion, seg_criterion, epoch)
    val_loss, val_depth_acc = val(model, val_loader, depth_criterion, seg_criterion)
    print('-' * 20)
    print('Epoch: {}\tTrain Loss: {:.6f}\tTrain Depth Acc: {:.6f}\tVal Loss: {:.6f}\tVal Depth Acc: {:.6f}'.format(
        epoch, train_loss, train_depth_acc, val_loss, val_depth_acc))
    train_losses.append(train_loss)
    train_depth_accs.append(train_depth_acc)
    val_losses.append(val_loss)
    val_depth_accs.append(val_depth_acc)
    lr_scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'model.pth')