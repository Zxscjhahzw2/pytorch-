import torch
import torchvision.datasets

from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NEURO_MODEL import *
from test1 import writer
from test2 import target

#GPU训练模式
device=torch.device("cuda")

#准备数据集
train_data=torchvision.datasets.CIFAR10("./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10("./dataset",train=True,download=True,transform=torchvision.transforms.ToTensor())

#length 长度
train_data_size=len(train_data)
test_data_size=len(test_data)

#如果train_data_size=10,训练数据集的长度为10
print(f"训练数据长度为：{train_data_size}")
print(f"测试数据长度为：{test_data_size}")

#利用dataloder来加载数据
train_dataloader=DataLoader(test_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#创建网络模型
#neuro=Neuro() 创建网络模型
#加载网络模型
neuro=torch.load("neuro_29time.pth")
neuro=neuro.to(device)
#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)

#优化器
learning_rate=0.01
optimizer=torch.optim.SGD(neuro.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step=0    #综训练次数
total_test_step=0   #总测试次数
epoch=10 #训练的轮数

#添加tensorboard
writer=SummaryWriter("./logs_train")

for i in range(epoch):
    print(f"----第{i+1}轮训练开始-----")

    neuro.train()
    #训练步骤开始
    for data in train_dataloader:
        imgs,targets=data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=neuro(imgs)
        loss=loss_fn(outputs,targets)
        #优化器调优
        optimizer.zero_grad()#优化之前梯度清0
        loss.backward()#损失反向传播
        optimizer.step()#对其中参数进行优化
        total_train_step+=1
        if total_train_step%100==0:
            print(f"训练次数为{total_train_step},loss:{loss.item()}")
            writer.add_scalar("tarin_loss",loss.item(),total_train_step)


#训练步骤开始
    neuro.eval()
    total_accuracy=0   #整体正确率
    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs=neuro(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy


            total_test_step+=1
    print(f"整体损失为{total_test_loss}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    print(f"整体正确率为{total_accuracy/test_data_size}")
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(neuro,"neuro_{}time.pth".format(i+30))
    print("模型已保存")


writer.close()



















































