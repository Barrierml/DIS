


# 定义一个双层卷积模块
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 定义一个上采样模块
def upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, 1)
    )

# 定义一个基于 ResNet18 和 U-Net 的图像分割网络
class ResUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # 加载预训练的 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        # 编码器部分
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        # 解码器部分
        # 使用 ResNet18 的最后一层作为中间层
        self.bottleneck = self.resnet.layer4
        # 进行跳跃连接，并使用双层卷积和上采样
        self.upsample4 = upsample(512, 256)
        self.decoder4 = double_conv(512, 256)
        self.upsample3 = upsample(256, 128)
        self.decoder3 = double_conv(256, 128)
        self.upsample2 = upsample(128, 64)
        self.decoder2 = double_conv(128, 64)
        self.upsample1 = upsample(64, 32)
        self.decoder1 = double_conv(64 + 32, 32)
        # 最后一层卷积，输出每个像素的类别概率
        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        # 编码器部分
        x1 = self.encoder1(x) # 得到第一层的特征图
        x2 = self.encoder2(x1) # 得到第二层的特征图
        x3 = self.encoder3(x2) # 得到第三层的特征图
        x4 = self.encoder4(x3) # 得到第四层的特征图
        print('x1.shape', x1.shape, 'x2.shape', x2.shape, 'x3.shape', x3.shape, 'x4.shape', x4.shape)
        # 中间层
        x_bottleneck = self.bottleneck(x4) # 得到中间层的特征图
        x_up_bottleneck = self.upsample4(x_bottleneck) # 对中间层的特征图进行上采样
        print('x_up_bottleneck:', x_up_bottleneck.shape)
        # 解码器部分，并进行跳跃连接
        x_up4 = torch.cat([x4, x_up_bottleneck], dim=1) # 将第四层和中间层上采样后的特征图拼接起来
        print('x_up4.shape:', x_up4.shape) # torch.Size([1, 768, 16, 16]) 
        x_out4 = self.decoder4(x_up4) # 得到第四层解码后的特征图
        print('x_out4.shape:', x_out4.shape) # torch.Size([1, 256, 16, 16])
        x_up3 = torch.cat([x3, self.upsample3(x_out4)], dim=1) # 将第三层和第四层解码后上采样后的特征图拼接起来
        x_out3 = self.decoder3(x_up3) # 得到第三层解码后的特征图

        x_up2 = torch.cat([x2, self.upsample2(x_out3)], dim=1) # 将第二层和第三层解码后上采样后的特征图拼接起来
        x_out2 = self.decoder2(x_up2) # 得到第二层解码后的特征图

        x_up1_1 = self.upsample1(x_out2) # 对第二层解码后的特征图进行上采样
        # print('x_up1_1.shape:', x_up1_1.shape) # torch.Size([1, 32, 256, 256])
        # print('x1.shape:', x1.shape) # torch.Size([1, 64, 256, 256])
        x_up1 = torch.cat([x1, x_up1_1], dim=1) # 将第一层和第二层解码后上采样后的特征图拼接起来
        # print('x_up1.shape:', x_up1.shape) # torch.Size([1, 64, 256, 256])
        x_out1 = self.decoder1(x_up1) # 得到第一层解码后的特征图

        # 最后一层卷积，输出每个像素的类别概率
        output = torch.sigmoid(self.conv_last(x_out1)) # 使用 sigmoid 函数将输出值映射到 [0, 1] 区间

        return output # 返回输出结果

# 测试模型输入与输出
model = ResUNet(n_class=1)
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(output.shape) # torch.Size([1, 1, 256, 256])

