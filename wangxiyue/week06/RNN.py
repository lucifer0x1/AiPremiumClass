# 作业
#1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。
#   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
#

import os
import shutil
import time

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST


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
    data_all=fetch_olivetti_faces(data_home='../data/face_data')

    img = data_all.images.reshape(-1,64,64)
    target = data_all.target
    # 归一化处理
    # mean = img.mean()
    # std = img.std()
    # img = (img - mean) / std
    return train_test_split(img,target,test_size=0.2,random_state=33,stratify=target,shuffle=True)

# torch DataLoader
def data_loader(batch_size,device):

    xTrain, xTest, yTrain, yTest = load_data_olivettiface()
    labels = set(yTrain)
    print('people label = ', len(labels))



    train_datasets = OlivettiDataset(xTrain,yTrain )
    test_datasets = OlivettiDataset(xTest, yTest)

    trainData = DataLoader(train_datasets, batch_size=batch_size, shuffle=True
                            ,generator=torch.Generator(device=device))
    testData = DataLoader(test_datasets, batch_size=batch_size, shuffle=False
                            ,generator=torch.Generator(device=device))
    return trainData, testData

#torch Dataset
class OlivettiDataset(Dataset):
    def __init__(self,x,y):
        super(OlivettiDataset,self).__init__()
        self.x = x
        self.y = y


    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        data = self.x[idx]
        target = self.y[idx]
        data =torch.tensor(data).float()
        target = torch.tensor(target).long()
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

def data_loader_mnist(batch_size,device):
    train = MNIST(root='../data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    test = MNIST(root='../data/MNIST', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True ,generator=torch.Generator(device=device))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False ,generator=torch.Generator(device=device))
    return train_loader,test_loader




class CustomTorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size , model_name):
        super(CustomTorchRNN, self).__init__()
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True , dropout=0.4)
            self.fc = nn.Linear(hidden_size, output_size)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True , dropout=0.4)
            self.fc = nn.Linear(hidden_size, output_size)
        else :
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.3,
                # nonlinearity='relu',
                bias=True,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output,end = self.rnn(x)
        y = self.fc(output[:, -1, :])
        # y = self.fc(end[-1,:,:])
        return y



###################################################################
###
###   public param
###
###################################################################
# check device
device = check_device()
torch.set_default_device(device)
############################# 超参 #################################
Learning_Rate = 0.001
Epochs = 50
BATCH_SIZE = 16
Momentum = 0.9
#



############################# 加载数据 ########################
#   b t c h w
#  weight = 64 ， height = 64

def train_mnist():
    ############################# 训练数据参数 #########################
    input_size = 28  # face: h = 64 , mnist: h= 28
    hidden_size = 50
    num_layers = 1
    output_size = 10  # face: 40个人 , mnist : 10 个数字

    total_train_step = 0  # 总体训练集次数
    total_test_step = 0  # 总体验证集次数

    ##
    model = CustomTorchRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           output_size=output_size)
    print('model initialized')
    Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=Momentum)

    train_data, test_data = data_loader_mnist(BATCH_SIZE, device)

    loss_fc = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(Epochs):
        model.train()
        train_loss = []
        for trainX, trainY in train_data:
            Optimizer.zero_grad()
            # print(trainX.shape,trainY.shape)
            n = trainX.shape[0]
            trainX = trainX.to(device)
            # print(trainX.shape)
            # trainX  = trainX.reshape(n, 28, 28)  # (batch,64,64)
            trainX = trainX.squeeze()
            # print(trainX.shape)
            trainY = trainY.to(device)
            # print(f'trainX.shape = {trainX.shape}')
            pred_train = model(trainX)
            # print('pred ==>',pred_train.shape,trainY.shape)
            loss = loss_fc(pred_train, trainY)
            loss.backward()
            Optimizer.step()
            train_loss.append(loss.item())
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(f'Epoch:{epoch + 1}/{Epochs} , 训练次数:{total_train_step} '
                      f', Loss:{loss.item():.4f}, 耗时: {(end_time - start_time):.1f} 秒')
                      # f', Loss:{np.average(train_loss):.6f}, 耗时: {(end_time - start_time):.1f} 秒')

        model.eval()
        acc = 0
        test_data_size = 0
        test_loss = []
        # 测试验证
        with (torch.no_grad()):
            for ii, (data, labels) in enumerate(test_data):  # 80
                # print(data.shape,labels.shape)
                n = data.shape[0]
                data = data.to(device).reshape(n, 28, 28)
                labels = labels.to(device)
                pred_test = model(data)
                loss_test = loss_fc(pred_test, labels)
                test_loss.append(loss_test.item())
                acc += (pred_test.argmax(1) == labels).sum().item()
                test_data_size += labels.size(0)
                writer.flush()
        print(
            f'acc={acc},test_data_size={test_data_size} | 测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc :{(acc / test_data_size * 100):.3f}% {color.END}')
        # writer.add_image('dont predict img',cant_img)
        writer.add_scalar('test_loss_avg', np.average(test_loss), total_test_step)
        writer.add_scalar('test_acc', acc / test_data_size, total_test_step)
        total_test_step += 1
        torch.save(model.state_dict(), f'../data/model/torch_rnn_module_{epoch + 1}.pth')
        writer.flush()

def train_face(model_name = 'RNN' , writer = None):
    ############################# 训练数据参数 #########################
    input_size = 64  # face: h = 64 , mnist: h= 28
    hidden_size = 128
    num_layers = 2
    output_size = 40  # face: 40个人 , mnist : 10 个数字
    total_train_step = 0  # 总体训练集次数
    total_test_step = 0  # 总体验证集次数

    ##
    model = CustomTorchRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, model_name =  model_name)
    model.to(device)
    print(f'{model_name} --> initialized')
    opt = torch.optim.AdamW(model.parameters(), lr=Learning_Rate, weight_decay = 1e-3 )

    train_data, test_data = data_loader(BATCH_SIZE,device)

    loss_fc = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(Epochs):
        model.train()
        train_loss = []
        for trainX , trainY in train_data:

            trainX = trainX.to(device) # (batch,64,64)
            trainY = trainY.to(device)
            pred_train = model(trainX)
            loss = loss_fc(pred_train, trainY)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(f'{model_name} --> Epoch:{epoch + 1}/{Epochs} , 训练次数:{total_train_step} '
                      f', Loss:{loss.item():.4f}, 耗时: {(end_time - start_time):.1f} 秒')
                # f', Loss:{np.average(train_loss):.6f}, 耗时: {(end_time - start_time):.1f} 秒')

        model.eval()
        acc = 0
        test_data_size = 0
        test_loss = []
        # 测试验证
        with (torch.no_grad()):
            for ii, (data, labels) in enumerate(test_data):  # 80
                # print(data.shape,labels.shape)
                labels = labels.to(device)
                pred_test = model(data)
                loss_test = loss_fc(pred_test, labels)
                test_loss.append(loss_test.item())
                acc += (pred_test.argmax(1) == labels).sum().item()
                test_data_size += labels.size(0)
        print(f'acc={acc},test_data_size={test_data_size} | 测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc :{(acc / test_data_size * 100):.3f}% {color.END}')
        # writer.add_image('dont predict img',cant_img)
        writer.add_scalar(f'test_loss_avg ',np.average(test_loss), total_test_step)
        writer.add_scalar(f'test_acc ', acc / test_data_size , total_test_step)
        total_test_step += 1
        # torch.save(model.state_dict(), f'../data/model/torch_rnn_module_{epoch + 1}.pth')
        writer.flush()

def now_time_str(now):
    return time.strftime('%Y-%m-%d %H:%M:%S', now)

if __name__ == '__main__':

    ############################# 配置 #################################
    start_time = time.localtime()
    LOG_DIR = 'logs_rnn' # ../data/logs_rnn
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)

    writer = SummaryWriter(LOG_DIR + '/' +now_time_str(start_time) +'RNN')
    train_face('RNN',writer)

    writer = SummaryWriter(LOG_DIR + '/' +now_time_str(start_time)  + 'GRU')
    train_face('GRU',writer)

    writer = SummaryWriter(LOG_DIR + '/' +now_time_str(start_time) + 'LSTM')
    train_face('LSTM',writer)

    # train_mnist()




    writer.close()