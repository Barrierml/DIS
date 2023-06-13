import torch
import torch.nn as nn
from models import DD_Net_Fuse, DD_Net_Single, DD_Net_RC
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
import time
import tqdm
import numpy as np

fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)
bce_loss = nn.BCELoss(size_average=True)

def im_preprocess(im,size):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    if(len(size)<2):
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = F.upsample(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor,0)

    return im_tensor.type(torch.uint8), im.shape[0:2]

class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, img_transform=None, gt_transform=None, name='train'):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.filenames = os.listdir(img_dir)
        self.base_cache_dir = '/root/autodl-tmp/dd-cache' + name
        # 将所有的图片提前处理并保存为 pt 文件, 保存到 cache 文件夹下
        for filename in tqdm.tqdm(self.filenames):
            # 创建 cache 文件夹
            if not os.path.exists(self.base_cache_dir):
                os.mkdir(self.base_cache_dir)
            if not os.path.exists(os.path.join(self.base_cache_dir, filename.replace('.jpg', '.pt'))):
                img = Image.open(os.path.join(self.img_dir, filename))
                if self.img_transform:
                    img = self.img_transform(img)
                torch.save(img, os.path.join(self.base_cache_dir, filename.replace('.jpg', '.pt')))
                # 将图片保存为 pt 文件提高读取速度
                gt = Image.open(os.path.join(self.gt_dir, filename.replace('.jpg', '.png')))
                if self.gt_transform:
                    gt = self.gt_transform(gt)
                torch.save(gt, os.path.join(self.base_cache_dir, filename.replace('.jpg', '_gt.pt')))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # 从 pt 文件中读取数据
        img = torch.load(os.path.join(self.base_cache_dir, self.filenames[idx].replace('.jpg', '.pt')))
        gt = torch.load(os.path.join(self.base_cache_dir, self.filenames[idx].replace('.jpg', '_gt.pt')))
        img = torch.divide(img,255.0)
        return img, gt

net = DD_Net_Single()
# 加载预训练好的模型
# net.load_state_dict(torch.load('./dd-net-ended.pth'))
# print('load pretrained model')
net.cuda()
print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

image_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])
# 图像分割处理
gt_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])

# 加载数据集
train_dataset = Image_Dataset('/root/DIS/ay-item-data/train/im', '/root/DIS/ay-item-data/train/gt', image_transform, gt_transform, name='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# 验证集
val_dataset = Image_Dataset('/root/DIS/ay-item-data/val/im', '/root/DIS/ay-item-data/val/gt', image_transform, gt_transform, name='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# 训练
def train(net, train_loader, optimizer, epoch, iter=0):
    net.train()
    average_loss = 0
    start_time = time.time()
    for i, (img, gt) in enumerate(train_loader):
        iter += 1
        img = img.cuda()
        gt = gt.cuda()
        pred = net(img)
        loss = net.loss(pred, gt)
        average_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印每个训练的 loss 与当前 epoch 的进度
        print('epoch: {}, iter: {}, percent: {:.2f}%, loss: {}, cost_time: {:.6f}'.format(epoch, iter, i / len(train_loader) * 100, loss.item(), time.time() - start_time))
        start_time = time.time()
    print('epoch: {}, average loss: {}'.format(epoch, average_loss / len(train_loader)))
    return average_loss / len(train_loader), iter

# 验证
def val(net, val_loader):
    net.eval()
    loss_sum = 0
    for j, (img, gt) in enumerate(val_loader):
        img = img.cuda()
        gt = gt.cuda()
        pred = net(img)
        loss = net.loss(pred, gt)
        loss_sum += loss.item()
    print('val loss: {}'.format(loss_sum / len(val_loader)))
    return loss_sum / len(val_loader)

def check_early_stopping(val_loss, patience=30):
    if len(val_loss) <= patience:
        return False
    best_val_loss = min(val_loss[:-patience])
    if min(val_loss) >= best_val_loss:
        return True 
    return False 

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

epochs = 10000000
current_epoch = 0

losses = []
val_losses = []
iter = 0

for epoch in range(current_epoch, epochs+1):
    train_loss, iter = train(net, train_loader, optimizer, epoch, iter=iter)
    losses.append(train_loss)
    current_epoch += 1
    v_loss = 0
    if epoch % 2 == 0:
        v_loss = val(net, val_loader)
        val_losses.append(v_loss)
    if epoch % 4 == 0:
        print('保存模型中')
        torch.save(net.state_dict(), './dd-net-{}-{}-{}-{:.6f}-{:.6f}-{:.6f}.pth'.format(epoch, net.__class__.__name__, iter, train_loss, v_loss, time.time()))
    if check_early_stopping(val_losses, 30):
        print("连续 30 epoch 没有任何提升，退出")
        torch.save(net.state_dict(), './dd-net-ended.pth')
        break

# 测试
def test(net, test_loader):
    net.eval()
    loss_sum = 0
    for j, (img, gt) in enumerate(test_loader):
        img = img.cuda()
        gt = gt.cuda()
        pred = net(img)
        loss = net.loss(pred, gt)
        loss_sum += loss.item()
    print('test loss: {}'.format(loss_sum / len(test_loader)))
    return loss_sum / len(test_loader)


# 测试集
# test_dataset = Image_Dataset('/root/DIS/DIS5K/DIS-TE/im', '/root/DIS/DIS5K/DIS-TE/gt', image_transform, gt_transform, name='test')
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# test(net, test_loader)