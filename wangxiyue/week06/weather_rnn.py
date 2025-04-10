#2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#   数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2
#
import csv
from datetime import datetime

import numpy as np
import torch
from torch import nn


def ymd2t(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.strftime("%Y%m%d"))

#
#STA、 Date, Precip ,WindGustSpd,MaxTemp,MinTemp,MeanTemp,Snowfall,PoorWeather,YR,MO,DA,PRCP,DR,SPD,MAX,MIN,MEA,SNF,SND,FT,FB,FTI,ITH,PGT,TSHDSBRSGF,SD3,RHX,RHN,RVG,WTE
#站号、日期、  降水量。  最大风 ， 最大温度，最小温度，平均温度 ， 降雪，
def load_weather_data(path):
    res  = {}

    with open(path,'r' , encoding = 'utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # [[time],[temp]]
            d  = ymd2t(row['Date'])
            res.setdefault(row['STA'],{})[d] = float(row['MaxTemp'])
    return res


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size,num_layers, output_size , model_name):
        drop = 0.3

        super(RNNModel, self).__init__()
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True , dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True ,dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)
        else : # 'RNN'
            self.rnn = nn.RNN( input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True ,dropout=drop)
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output,end = self.rnn(x)
        y = self.fc(output[:, -1, :])
        # y = self.fc(end[-1,:,:])
        return y

if __name__ == '__main__':

    # b t feature
    #
    res = load_weather_data('../data/SummaryofWeather.csv')
    sta_num = len(res) # 159个站点

    # 输入数据
    # 双通道， 通道1 ： 时间date， 通道2 最大温maxTemp

    # input
    batch_size = 64  # length
    time_step = 10  # 时间窗口
    pred_step = 5  # 预测未来 窗口

    chanel = 3
    # 站点数量 就是总 数据集
    X = torch.randn(sta_num,time_step,3)
    Y = torch.randn(sta_num,pred_step)

    for i , sta in enumerate(res):
        ymd = np.array([k for k in res[sta].keys()])
        tmp =np.array([v for v in res[sta].values()])

        ll = len(ymd)

        # x_seq [time_step , chanel]
        single_station_all_time_x = torch.randn(ll,time_step,3)
        single_station_all_time_y = torch.randn(ll,pred_step)
        for i in range(ll - time_step - pred_step + 1):
            s = np.full(time_step,int(sta)) # station
            x_seq = torch.tensor(np.array([ymd[i:i + time_step], tmp[i:i + time_step] , s]))
            # c,t --> t,c  (10,2)
            x_seq = x_seq.permute(1, 0)
            y_seq = torch.tensor(tmp[i + time_step:i + time_step + pred_step])  # 5

        single_station_all_time_x

    print(X.shape)
    print(Y.shape)



