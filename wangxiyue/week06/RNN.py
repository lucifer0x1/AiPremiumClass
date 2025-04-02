# 作业
#1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。
#   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
#
#2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#   数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2
#
import os
import shutil
import time

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.functional import max_pool1d
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from wangxiyue.week03.torch_nn_full import optimizer


def check_device():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'use {device} ')
    return device

###olivetti###
# data (400,4096)
# target(400,)
# images(400,64,64)
# 40个人
def load_data_olivettiface():
    # data,img,target,desc = fetch_olivetti_faces(data_home='../data/face_data',return_X_y=False)
    data_all =  fetch_olivetti_faces(data_home='../data/face_data', return_X_y=False)

    data = data_all["data"]
    img = data_all["images"]
    target = data_all["target"]
    return train_test_split(data,target,img,test_size=0.2,random_state=42,shuffle=True)

# torch DataLoader
def data_loader(batch_size,device):
    transform_train = transforms.Compose([
        transforms.RandomRotation(16),
        transforms.RandomAffine(7, translate=(0.11, 0.13), shear=0.16),
        transforms.ToTensor()
    ])
    ######
    transform_train = None
    ######

    xTrain, xTest, yTrain, yTest,imgTrain,imgTest = load_data_olivettiface()
    labels = set(yTrain)
    print('people label = ', len(labels))
    # print('load Img ==> ',imgTrain.shape,imgTest.shape)
    train_datasets = OlivettiDataset(xTrain,yTrain , transform=transform_train)
    test_datasets = OlivettiDataset(xTest, yTest , img = imgTest,toTensor=False)

    trainData = DataLoader(train_datasets, batch_size=batch_size, shuffle=True
                            ,generator=torch.Generator(device=device))
    testData = DataLoader(test_datasets, batch_size=batch_size, shuffle=True
                            ,generator=torch.Generator(device=device))
    return trainData, testData

#torch Dataset
class OlivettiDataset(Dataset):
    def __init__(self,x,y,img = None,transform=None , toTensor=True):
        # super(OlivettiDataset,self).__init__()
        self.x = x
        self.y = y
        self.toTensor = toTensor
        self.img = img
        if self.toTensor==False:
            print('self.toTensor==False  x.shape ==> ',x.shape)
        if transform is not None:
            self.transform = transform
        else :
            self.transform = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        data = self.x[idx]
        target = self.y[idx]
        img = None
        if self.img is not None:
            img = self.img[idx]

        if self.transform is not None:
            data = self.transform(data[idx])
            target=self.transform(target[idx])
        if self.toTensor:
            data =torch.tensor(data)
            target = torch.tensor(target)
            # if self.img is not None:
            #     img = torch.tensor(img)
        if self.img is not None:
            return data, target, img
        return data, target

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class CustomTorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(CustomTorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = nn.Tanh()(x)
        output,end = self.rnn(x)
        # end = nn.Tanh()(end)
        # y = self.fc(output[:,-1,:])
        y = self.fc(end[-1,:,:])
        return y

if __name__ == '__main__':

    # check device
    device = check_device()
    torch.set_default_device(device)


    ############################# 超参 #################################
    Learning_Rate = 0.005
    Epochs = 10
    BATCH_SIZE = 64
    Momentum = 0.9
    #
    input_size = 64
    hidden_size = 50
    num_layers = 5
    output_size = 40 # 40个人
    model = CustomTorchRNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, output_size=output_size)
    print('model initialized')
    Optimizer = torch.optim.AdamW(model.parameters(),lr=Learning_Rate)
    ############################# 配置 #################################
    LOG_DIR = '../data/logs_olivetti'
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)
    start_time = time.time()
    ############################# 训练数据参数 #########################
    total_train_step = 0  # 总体训练集次数
    total_test_step = 0  # 总体验证集次数
    ############################# 加载数据 ########################
    #   b t c h w
    #  weight = 64 ， height = 64
    train_data, test_data = data_loader(BATCH_SIZE,device)
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    loss_fc = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(Epochs):
        model.train()
        train_loss = []
        for trainX , trainY in train_data:
            optimizer.zero_grad()
            # print(trainX,trainY.shape)
            n = trainX.shape[0]
            trainX = trainX.to(device).reshape(n,64,64)
            trainY = trainY.to(device)
            pred_train = model(trainX).squeeze()
            # print('pred ==>',pred_train.shape,trainY.shape)
            loss = loss_fc(pred_train, trainY)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            total_train_step += 1
            if total_train_step % 10 == 0:
                end_time = time.time()
                print(f'Epoch:{epoch + 1}/{Epochs} , 训练次数:{total_train_step} '
                      f', Loss:{np.average(train_loss):.6f}, 耗时: {(end_time - start_time):.1f} 秒')

        model.eval()
        acc = 0
        test_data_size = 0
        test_loss = []
        cant_img = []
        # 测试验证
        with (torch.no_grad()):
            for ii, (data, labels, img) in enumerate(test_data):  # 80
                # print(data.shape,labels.shape)
                n = data.shape[0]
                data = data.to(device).reshape(n,64,64)
                labels = labels.to(device)
                pred_test = model(data).squeeze()
                loss_test = loss_fc(pred_test, labels)
                test_loss.append(loss_test.item())
                acc += (pred_test.argmax(1) == labels).sum().item()
                test_data_size += labels.size(0)
                # 获取为识别的照片
                cant_img += img[labels != pred_test.argmax(1)]
                # 下面这个取反的逻辑错误
                ### cant_img +=img[~torch.isin(labels,pred_test.argmax(1))]
            for index, imt in enumerate(cant_img):  # bach size 60
                writer.add_image(f'Un_Pred_Img/epoch:{epoch + 1}/{index + 1}.img', imt,
                                 dataformats='HW', global_step=index + 1)
                writer.flush()
        print(f'acc={acc},test_data_size={test_data_size} | 测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc :{(acc / test_data_size * 100):.3f}% {color.END}')
        # writer.add_image('dont predict img',cant_img)
        writer.add_scalar('test_loss_avg', np.average(test_loss), total_test_step)
        writer.add_scalar('test_acc', acc / test_data_size, total_test_step)
        total_test_step += 1
        torch.save(model.state_dict(), f'../data/model/torch_rnn_module_{epoch + 1}.pth')
        writer.flush()
writer.close()
