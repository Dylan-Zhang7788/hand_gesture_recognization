from EarlyStopping import EarlyStopping
import hiddenlayer as hl

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from net.MyVGG import MyVggModel
# 导入预训练好的VGG16网络
vgg16 = models.vgg16(pretrained=True)

vgg = vgg16.features
for param in vgg.parameters():
    param.requires_grad_(False)

# 使用VGG16的特征提取层＋新的全连接层组成新的网络

# 定义网络结构
Myvggc = MyVggModel().cuda()
print(Myvggc)

early_stopping = EarlyStopping(patience=15, verbose=True)


# 定义transform

train_data_transforms = transforms.Compose([
    transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
    # 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取图像
train_data_dir = "./dataset/train_VGG"
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
train_data_loader = Data.DataLoader(train_data, batch_size=16,
                                    shuffle=True, num_workers=1)

print("训练集样本数:", len(train_data.targets))


# 定义优化器
optimizer = torch.optim.Adam(Myvggc.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss().cuda()   # 损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
# 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(1500):
    train_loss_epoch = 0
    train_corrects = 0

    # 对训练数据的迭代器进行迭代计算
    Myvggc.train()
    for step, (b_x, b_y) in enumerate(train_data_loader):
        # 计算每个batch的
        b_x, b_y = b_x.cuda(), b_y.cuda()
        output = Myvggc(b_x)            # CNN在训练batch上的输出
        loss = loss_func(output, b_y)   # 交叉熵损失函数
        pre_lab = torch.argmax(output, 1)
        optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
        loss.backward()                 # 损失的后向传播，计算梯度
        optimizer.step()                # 使用梯度进行优化
        train_loss_epoch += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab == b_y.data)
    # 计算一个epoch的损失和精度
    train_loss = train_loss_epoch / len(train_data.targets)
    train_acc = train_corrects.double() / len(train_data.targets)

    # 保存每个epoch上的输出loss和acc
    history1.log(epoch, train_loss=train_loss,
                 train_acc=train_acc.item(),
                 )
    # 可视网络训练的过程
    with canvas1:
        canvas1.draw_plot(history1["train_loss"])
        canvas1.draw_plot(history1["train_acc"])

    if early_stopping(train_loss, Myvggc) is True:
        break

    # 保存模型
torch.save(Myvggc, "./checkpoints/VGG16/VGG_trained.pkl")
